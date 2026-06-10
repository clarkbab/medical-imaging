from dicomset import config
from augmed import Pipeline, RandomAffine
from dicomset.utils import fov_centre, logger
from dicomset.dicom.utils import from_rtplan_dicom
from dicomset.nifti.utils import load_dataset as load_nifti_dataset
from dicomset.training import TrainingDataset
from dicomset.typing import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time
import torch
from tqdm import tqdm
from typing import *
import wandb

from types import SimpleNamespace

from mymi.loaders.mlm import DRRLoader
from mymi.models.architectures.cgan.pix2pix2d import Pix2PixModel
from mymi.training.utils.lr_find import run_lr_find_pix2pix
from mymi.utils.cdog import load_raw_frame
from mymi.utils.interval import interval_matches

def train_pix2pix(
    dataset: str,
    pat: PatientID,
    project: str,
    model: str,
    n_epochs: int,
    lr_init: float,
    n_train_angles: int | None = None,
    n_val_angles: int | None = None,
    n_val_volumes: int | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    preload_val_data: bool = False,
    beta1: float = 0.5,
    gan_mode: str = 'vanilla',
    input_nc: int = 1,
    lambda_L1: float = 100.0,
    log_images: bool = False,
    log_images_local: bool = True,
    n_train_images: int = 5,
    n_val_images: int = 5,
    train_image_interval: str = 'epoch:5',
    lr_find: bool = False,
    lr_find_min_lr: float = 1e-7,
    lr_find_max_lr: float = 1,
    lr_find_n_iter: int = 100,
    n_layers_D: int = 3,
    ndf: int = 64,
    netD: str = 'basic',
    netG: str = 'unet_256',
    ngf: int = 64,
    no_dropout: bool = False,
    norm: str = 'batch',
    output_nc: int = 1,
    random_seed: int = 42,
    resume: bool = False,
    resume_ckpt: str = 'last',
    use_logger: bool = True,
    val_image_interval: str = 'epoch:5',
    ) -> None:
    logger.log_method('Training VALKIM pix2pix model (DRR)')
    model_name = model  # Use model for actual model.

    # Set seed for reproducible runs.
    torch.manual_seed(random_seed)

    ckpt_path = os.path.join(config.dirs.models, project, model_name)
    if os.path.exists(ckpt_path) and not resume and not lr_find:
        # Clean up old run files.
        logger.info(f"Removing old checkpoint directory {ckpt_path}.")
        shutil.rmtree(ckpt_path)
    os.makedirs(ckpt_path, exist_ok=True)

    # Create augmentation.
    transform_train = Pipeline([
        RandomAffine(r=10, s=[0.8, 1.2], t=20),
    ])
    print(transform_train)

    # Load projection geometry.
    info = load_nifti_dataset('VALKIM-PP').params['patient-info']

    filepath = info[pat]['treatment-image']
    _, tiff_info = load_raw_frame(filepath)

    filepath = info[pat]['rtplan']
    plan_info = from_rtplan_dicom(filepath)

    # Set projection parameters.
    isocentre = plan_info['isocentre']
    sid = tiff_info['sid']
    sdd = tiff_info['sdd']
    det_size = tiff_info['det-size']
    det_spacing = tiff_info['det-spacing']
    det_offset = tiff_info['det-offset']
    print(f"Projection geometry: isocentre={isocentre}, sid={sid}, sdd={sdd}, det_size={det_size}, det_spacing={det_spacing}, det_offset={det_offset}")
    geometry = dict(
        isocentre=isocentre,
        sid=sid,
        sdd=sdd,
        det_size=det_size,
        det_spacing=det_spacing,
        det_offset=det_offset,
    )

    # Create data loaders.
    loader_kwargs = dict(
        batch_size=batch_size,
        n_train_angles=n_train_angles,
        n_val_angles=n_val_angles,
        n_val_volumes=n_val_volumes,
        num_workers=num_workers,
        preload_val_data=preload_val_data,
        projection_geometry=geometry,
        transform_train=transform_train,
    )
    tl, vl = DRRLoader.build_loaders(dataset, pat, **loader_kwargs)
    print(f'[train] train_loader_len={len(tl)} val_loader_len={len(vl)} batch_size={batch_size} num_workers={num_workers} preload_val={preload_val_data}')

    # Create model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"Expected CUDA device, got '{device}'. Ensure a GPU is available."
    gpu_ids = [0] if torch.cuda.is_available() else []
    opt = SimpleNamespace(
        gpu_ids=gpu_ids,
        isTrain=True,
        checkpoints_dir=config.dirs.models,
        name=os.path.join(project, model_name),
        direction='AtoB',
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        netG=netG,
        norm=norm,
        no_dropout=no_dropout,
        init_type='normal',
        init_gain=0.02,
        ndf=ndf,
        netD=netD,
        n_layers_D=n_layers_D,
        gan_mode=gan_mode,
        lr=lr_init,
        beta1=beta1,
        lambda_L1=lambda_L1,
    )
    model = Pix2PixModel(opt)
    ckpt_info = {}
    if resume:
        logger.info(f"Restoring model from checkpoint '{resume_ckpt}'.")
        filepath = os.path.join(ckpt_path, f'{resume_ckpt}.ckpt')
        ckpt_info = torch.load(filepath, map_location=device)
        model.netG.load_state_dict(ckpt_info['netG'])
        model.netD.load_state_dict(ckpt_info['netD'])
        model.optimizer_G.load_state_dict(ckpt_info['optimizer_G'])
        model.optimizer_D.load_state_dict(ckpt_info['optimizer_D'])

    n_params_G = sum(p.numel() for p in model.netG.parameters())
    n_params_D = sum(p.numel() for p in model.netD.parameters())
    print(f'[model] generator params={n_params_G:,}  discriminator params={n_params_D:,}')

    # --- LR Find ---
    if lr_find:
        run_lr_find_pix2pix(
            model=model,
            train_loader=tl,
            device=device,
            dataset=dataset,
            project=project,
            model_name=model_name,
            min_lr=lr_find_min_lr,
            max_lr=lr_find_max_lr,
            n_iter=lr_find_n_iter,
            output_nc=output_nc,
        )
        return

    # Set up logger.
    run_dir = os.path.join(config.dirs.runs, project, model_name)
    if use_logger:
        run = wandb.init(
            dir=run_dir,
            entity='clarkbab',
            project=project,
            name=model_name,
        )

    train_image_save_dir = os.path.join(run_dir, 'images', 'train')
    val_image_save_dir = os.path.join(run_dir, 'images', 'val')
    if log_images_local:
        os.makedirs(train_image_save_dir, exist_ok=True)
        os.makedirs(val_image_save_dir, exist_ok=True)

    # Checkpoints are saved at the end of the epoch/step, so need to increment values by 1.
    start_epoch = ckpt_info['epoch'] + 1 if resume else 0
    step = ckpt_info['step'] + 1 if resume else 0
    val_losses = ckpt_info['val-losses'] if resume else [] # Calculating moving average for checkpointing - less noisy.
    min_val_loss = ckpt_info['min-val-loss'] if resume else np.inf
    val_loss_smoothing = 1
    n_best_ckpts = 1
    best_ckpts = ckpt_info['best-ckpts'] if resume else []
    for e in range(start_epoch, n_epochs):
        # Training loop.
        model.netG.train()
        model.netD.train()
        # Create projections from augmented volumes at the start of each epoch.
        tl.dataset.create_projections(e)
        train_iter = iter(tl)
        for i, (xs, ys, angles) in enumerate(tqdm(train_iter, desc=f'Epoch {e}/{n_epochs} (train)', leave=False, total=len(tl))):
            batch_start = time.perf_counter()
            if i == 0:
                print(f'[train] first batch shapes xs={tuple(xs.shape)} ys={tuple(ys.shape)} angles_type={type(angles).__name__}')

            convert_start = time.perf_counter()
            input_dict = {'A': xs.to(device), 'B': ys[:, :output_nc].to(device), 'A_paths': [], 'B_paths': []}
            convert_time = time.perf_counter() - convert_start

            set_input_start = time.perf_counter()
            model.set_input(input_dict)
            set_input_time = time.perf_counter() - set_input_start

            opt_start = time.perf_counter()
            model.optimize_parameters()
            opt_time = time.perf_counter() - opt_start

            losses_start = time.perf_counter()
            losses = model.get_current_losses()
            losses_time = time.perf_counter() - losses_start

            if i == 0:
                print(f'[train] first batch timing convert={convert_time:.3f}s set_input={set_input_time:.3f}s optimize={opt_time:.3f}s losses={losses_time:.3f}s total={time.perf_counter()-batch_start:.3f}s')

            # Record metrics to Wandb.
            if use_logger:
                run.log({
                    'epoch': e,
                    'step': step,
                    **{f'train/{k}': v for k, v in losses.items()},
                }, step=step)

            # Log train images.
            if log_images and interval_matches(step, train_image_interval, len(tl), step_match_length=n_train_images):
                x_slice = model.real_A[0, 0].cpu().numpy()
                y_slice = model.real_B[0, 0].cpu().numpy()
                y_pred_slice = model.fake_B[0, 0].detach().cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(x_slice.T, cmap='gray')
                axes[0].set_title('Input')
                axes[0].axis('off')
                axes[1].imshow(y_slice.T, cmap='gray', vmin=0, vmax=1)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                axes[2].imshow(y_pred_slice.T, cmap='gray', vmin=0, vmax=1)
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                fig.tight_layout()

                filename = f'epoch-{e:04d}-step-{step:06d}-train-{i:03d}.png'
                if log_images_local:
                    fig.savefig(os.path.join(train_image_save_dir, filename), dpi=150, bbox_inches='tight')
                else:
                    run.log({'train/image': wandb.Image(fig)}, step=step)
                plt.close(fig)

            # Increment training step.
            step += 1

        # Validation loop.
        model.eval()
        val_iter = iter(vl)
        epoch_val_losses = []
        if e == start_epoch:
            print(f'[train] validation_loader_len={len(vl)}')
        for i, (xs, ys, angles) in enumerate(tqdm(val_iter, desc=f'Epoch {e}/{n_epochs} (val)', leave=False, total=len(vl))):
            if i == 0:
                print(f'[train] first val batch shapes xs={tuple(xs.shape)} ys={tuple(ys.shape)}')
            input_dict = {'A': xs.to(device), 'B': ys[:, :output_nc].to(device), 'A_paths': [], 'B_paths': []}
            model.set_input(input_dict)
            model.test()
            val_l1_loss = model.criterionL1(model.fake_B, model.real_B).item() * lambda_L1

            epoch_val_losses += [val_l1_loss]

            # Log images.
            if log_images and interval_matches(step, val_image_interval, len(val_iter), step_match_length=n_val_images):
                regions = ['GTV']
                for j, r in enumerate(regions):
                    # Log first batch item only.
                    x_slice = model.real_A[0, 0].cpu().numpy()
                    y_slice = model.real_B[0, j].cpu().numpy()
                    y_pred_slice = model.fake_B[0, j].detach().cpu().numpy()

                    # Get centre of extent of ground truth.
                    centre_vox = fov_centre(y_slice.shape)
                    if centre_vox is None:
                        continue

                    title = f'region:{r}'
                    masks = {
                        'ground_truth': {
                            'mask_data': y_slice,
                            'class_labels': regions,
                        },
                        'predictions': {
                            'mask_data': y_pred_slice,
                            'class_labels': regions,
                        }
                    }

                    if log_images_local:
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                        axes[0].imshow(x_slice.T, cmap='gray')
                        axes[0].set_title('Input')
                        axes[0].axis('off')
                        axes[1].imshow(y_slice.T, cmap='gray', vmin=0, vmax=1)
                        axes[1].set_title(f'Ground Truth ({r})')
                        axes[1].axis('off')
                        axes[2].imshow(y_pred_slice.T, cmap='gray', vmin=0, vmax=1)
                        axes[2].set_title(f'Prediction ({r})')
                        axes[2].axis('off')
                        fig.tight_layout()
                        filename = f'epoch-{e:04d}-step-{step:06d}-{r}.png'
                        fig.savefig(os.path.join(val_image_save_dir, filename), dpi=150, bbox_inches='tight')
                        plt.close(fig)
                    else:
                        run.log({
                            title: wandb.Image(
                                x_slice.T,
                                masks=masks,
                            )
                        }, step=step)

        # Save mean validation loss.
        mean_val_loss = np.mean(epoch_val_losses)
        if use_logger:
            run.log({ 'val/loss-epoch-mean': mean_val_loss }, step=step)
        val_losses.append(mean_val_loss)

        # Save best model/s.
        if len(val_losses) >= val_loss_smoothing:
            smoothed_val_loss = np.mean(val_losses[-val_loss_smoothing:])
            if use_logger:
                run.log({ 'val/loss-ma': smoothed_val_loss }, step=step)
            if smoothed_val_loss < min_val_loss:
                min_val_loss = smoothed_val_loss

                # Update 'best_ckpts'.
                ckpt = f'loss={min_val_loss:.6f}_epoch={e}_step={step}'
                best_ckpts.insert(0, ckpt)
                if len(best_ckpts) > n_best_ckpts:
                    old_ckpt = best_ckpts.pop()
                    filepath = os.path.join(ckpt_path, f'{old_ckpt}.ckpt')
                    os.remove(filepath)

                # Save model.
                ckpt_data = {
                    'best-ckpts': best_ckpts,
                    'epoch': e,
                    'min-val-loss': min_val_loss,
                    'netG': model.netG.state_dict(),
                    'netD': model.netD.state_dict(),
                    'optimizer_G': model.optimizer_G.state_dict(),
                    'optimizer_D': model.optimizer_D.state_dict(),
                    'step': step,
                    'val-losses': val_losses,     # Required for moving average.
                }
                filepath = os.path.join(ckpt_path, f'{ckpt}.ckpt')
                torch.save(ckpt_data, filepath)
                filepath = os.path.join(ckpt_path, 'best.ckpt')
                torch.save(ckpt_data, filepath)

        # Save current model (last + per-epoch).
        ckpt_data = {
            'best-ckpts': best_ckpts,
            'epoch': e,
            'min-val-loss': min_val_loss,
            'netG': model.netG.state_dict(),
            'netD': model.netD.state_dict(),
            'optimizer_G': model.optimizer_G.state_dict(),
            'optimizer_D': model.optimizer_D.state_dict(),
            'step': step,
            'val-losses': val_losses,     # Required for moving average.
        }
        filepath = os.path.join(ckpt_path, 'last.ckpt')
        torch.save(ckpt_data, filepath)
        filepath = os.path.join(ckpt_path, f'epoch={e:04d}.ckpt')
        torch.save(ckpt_data, filepath)

if __name__ == '__main__':
    pat_id = 'PAT1'
    version = '001'
    train_pix2pix(
        dataset='VALKIM-PP',
        pat=pat_id,
        project='MLM-VALKIM',
        model=f'pix2pix2d-{pat_id}-{version}',
        n_epochs=100,
        lr_init=5e-5,
        batch_size=32,
        num_workers=4,
        preload_val_data=False,
        beta1=0.5,
        gan_mode='vanilla',
        input_nc=1,
        lambda_L1=100.0,
        log_images=True,
        log_images_local=True,
        lr_find=False,
        lr_find_min_lr=1e-7,
        lr_find_max_lr=1,
        lr_find_n_iter=100,
        n_train_angles=361,
        n_val_angles=100,
        n_val_volumes=10,
        n_layers_D=3,
        ndf=64,
        netD='basic',
        netG='unet_256',
        ngf=64,
        no_dropout=False,
        norm='batch',
        output_nc=1,
        random_seed=42,
        resume=False,
        use_logger=True,
    )
