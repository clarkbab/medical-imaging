from augmed import Pipeline, RandomAffine
from dicomset import config
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
import torch
from tqdm import tqdm
from typing import *
import wandb

from types import SimpleNamespace

from mymi.loaders.mlm import StaticLoader
from mymi.models.architectures.cgan.pix2pix2d import Pix2PixModel
from mymi.training.utils.lr_find import run_lr_find_pix2pix
from mymi.utils.cdog import load_raw_frame
from mymi.utils.interval import interval_matches

def train_pix2pix_static(
    dataset: str,
    pat: PatientID,
    project: str,
    model: str,
    n_epochs: int,
    lr_init: float,
    n_train_angles: int | None = None,
    n_train_volumes: int | None = None,
    n_val_angles: int | None = None,
    n_val_volumes: int | None = None,
    batch_size: int = 32,
    beta1: float = 0.5,
    gan_mode: str = 'vanilla',
    input_nc: int = 1,
    lambda_L1: float = 100.0,
    log_images: bool = False,
    log_images_local: bool = True,
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
    logger.log_method('Training VALKIM pix2pix model')
    model_name = model  # Use model for actual model.

    # Set seed for reproducible runs.
    torch.manual_seed(random_seed)

    ckpt_path = os.path.join(config.dirs.models, project, model_name)
    if os.path.exists(ckpt_path) and not resume and not lr_find:
        # Clean up old run files.
        logger.info(f"Removing old checkpoint directory {ckpt_path}.")
        shutil.rmtree(ckpt_path)
    os.makedirs(ckpt_path, exist_ok=True)

    # # Create augmentation.
    # transform_train = Pipeline([
    #     RandomAffine(r=10, s=[0.8, 1.2], t=20),
    # ])
    # print(transform_train)

    # Load projection geometry.
    info = load_nifti_dataset('VALKIM-PP').params['patient-info']

    # Does this change between fractions??
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
        n_train_volumes=n_train_volumes,
        n_val_angles=n_val_angles,
        n_val_volumes=n_val_volumes,
        # transform_train=transform_train,
    )
    tl, vl = StaticLoader.build_loaders(dataset, pat, **loader_kwargs)

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
    if use_logger:
        run = wandb.init(
            dir=config.dirs.reports,
            entity="clarkbab",
            project=project,
            name=model_name,
        )

    # Set up local image save directory.
    image_save_dir = os.path.join(config.dirs.runs, project, model_name, 'images')
    if log_images_local:
        os.makedirs(image_save_dir, exist_ok=True)

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
        # tl.dataset.create_projections(e)
        train_iter = iter(tl)
        for xs, ys, angles in tqdm(train_iter, desc=f'Epoch {e}/{n_epochs} (train)', leave=False):
            input_dict = {'A': xs.to(device), 'B': ys[:, :output_nc].to(device), 'A_paths': [], 'B_paths': []}
            model.set_input(input_dict)
            model.optimize_parameters()
            losses = model.get_current_losses()

            # Record metrics to Wandb.
            if use_logger:
                run.log({
                    'epoch': e,
                    'step': step,
                    **{f'train/{k}': v for k, v in losses.items()},
                }, step=step)

            # Increment training step.
            step += 1

        # Validation loop.
        model.eval()
        val_iter = iter(vl)
        epoch_val_losses = []
        for xs, ys, angles in tqdm(val_iter, desc=f'Epoch {e}/{n_epochs} (val)', leave=False):
            input_dict = {'A': xs.to(device), 'B': ys[:, :output_nc].to(device), 'A_paths': [], 'B_paths': []}
            model.set_input(input_dict)
            model.test()
            val_l1_loss = model.criterionL1(model.fake_B, model.real_B).item() * lambda_L1

            # Record checkpointing metric.
            epoch_val_losses += [val_l1_loss]

            # Record metrics.
            if use_logger:
                run.log({
                    'epoch': e,
                    'step': step,
                    'val/loss_G_L1': val_l1_loss,
                }, step=step)

            # Log images.
            if log_images and interval_matches(step, val_image_interval, len(val_iter)):
                regions = ['GTV']
                for i, r in enumerate(regions):
                    c = i + 1

                    # Log first batch item only.
                    x_r = model.real_A[0].cpu().numpy()
                    y_r = model.real_B[0, i].cpu().numpy()
                    y_pred_r = model.fake_B[0, i].cpu().numpy()

                    # Get centre of extent of ground truth.
                    centre_vox = fov_centre(y_r)
                    if centre_vox is None:
                        # Presumably data augmentation has pushed the label out of view.
                        continue

                    # Plot each orientation.
                    for a, c in enumerate(centre_vox):
                        # Get 2D slice.
                        indices = tuple([c if k == a else slice(0, x_r.shape[i]) for k in range(3)])
                        y_pred_slice, x_slice, y_slice = y_pred_r[indices], x_r[indices], y_r[indices]

                        # Fix orientation.
                        if a in (0, 1):     # Sagittal/coronal views.
                            y_pred_slice = np.rot90(y_pred_slice)
                            x_slice = np.rot90(x_slice)
                            y_slice = np.rot90(y_slice)
                        else:               # Axial view.
                            x_slice = np.transpose(x_slice)
                            y_slice = np.transpose(y_slice)
                            y_pred_slice = np.transpose(y_pred_slice)

                        title = f'region:{r}:axis:{a}'
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
                            axes[0].imshow(x_slice, cmap='gray')
                            axes[0].set_title('Input')
                            axes[0].axis('off')
                            axes[1].imshow(y_slice, cmap='gray', vmin=0, vmax=1)
                            axes[1].set_title(f'Ground Truth ({r})')
                            axes[1].axis('off')
                            axes[2].imshow(y_pred_slice, cmap='gray', vmin=0, vmax=1)
                            axes[2].set_title(f'Prediction ({r})')
                            axes[2].axis('off')
                            fig.tight_layout()
                            filename = f'epoch-{e:04d}-step-{step:06d}-{r}-axis{a}.png'
                            fig.savefig(os.path.join(image_save_dir, filename), dpi=150, bbox_inches='tight')
                            plt.close(fig)
                        else:
                            run.log({
                                title: wandb.Image(
                                    x_slice,
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
                print(best_ckpts)
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

        # Save current model.
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

if __name__ == '__main__':
    train_pix2pix_static(
        dataset='VALKIM-PP-STATIC',
        pat='PAT1',
        project='MLM-VALKIM',
        model='pix2pix2d-static',
        n_epochs=100,
        lr_init=5e-5,   # Based on LR find result.
        batch_size=32,
        beta1=0.5,
        gan_mode='vanilla',
        input_nc=1,
        lambda_L1=100.0,
        log_images=True,
        log_images_local=True,
        lr_find=True,
        lr_find_min_lr=1e-7,
        lr_find_max_lr=1,
        lr_find_n_iter=100,
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
        use_logger=False,
    )
