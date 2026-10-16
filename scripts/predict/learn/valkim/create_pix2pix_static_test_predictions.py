import dicomset as ds
from dicomset import config
from dicomset.utils import arg_to_list
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from types import SimpleNamespace

from mymi.models.architectures.architectures import get_model
from mymi.utils.cdog import list_tiff_arcs, list_tiff_fractions

project = 'MLM-VALKIM'
model_base = 'pix2pix2d-static'
model_v = 10
arch = 'pix2pix:unet_256'
# epoch = 0
# epochs = [epoch]
# epochs = [0, 5, 10, 20, 50]
epochs = list(range(100))
epochs = [0]

tdataset = 'VALKIM-PP-STATIC'
tset = ds.load(tdataset, 'training')
pat_ids = [1, 2, 3, 4]
pat_ids = [1]
fractions = 'all'
fractions = [1, 2, 3]
fractions = [1]
arcs = 'all'
# arcs = [1, 2, 3]
# arcs = [1]
# arcs = [2, 3]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in tqdm(epochs, desc='Epochs'):
    ckpt = f'epoch={epoch:04d}'

    for p in tqdm(pat_ids, leave=False):
        model_name = f'{model_base}-PAT{p}-{model_v:03d}'

        # Load full Pix2PixModel for inference; strip DataParallel 'module.' prefix from checkpoint.
        opt = SimpleNamespace(
            gpu_ids=[],
            isTrain=False,
            checkpoints_dir=config.dirs.models,
            name=os.path.join(project, model_name),
            direction='AtoB',
            input_nc=1,
            output_nc=1,
            ngf=64,
            netG='unet_256',
            norm='batch',
            no_dropout=False,
            init_type='normal',
            init_gain=0.02,
        )
        model = get_model(arch, n_output_channels=1, opt=opt)
        ckpt_filepath = os.path.join(config.dirs.models, project, model_name, f'{ckpt}.ckpt')
        ckpt_data = torch.load(ckpt_filepath, map_location=device)
        state_dict = {k.removeprefix('module.'): v for k, v in ckpt_data['netG'].items()}
        model.netG.load_state_dict(state_dict)
        model.netG.to(device)
        model.netG.eval()
        print(f"Loaded {model_name} from epoch {epoch}")

        patpath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient{p:02d}"
        pat_fractions = arg_to_list(fractions, int, literals={'all': list_tiff_fractions(patpath)})

        for f in tqdm(pat_fractions, leave=False):
            fracpath = os.path.join(patpath, f"Fx{f:02d}")
            pat_arcs = arg_to_list(arcs, int, literals={'all': list_tiff_arcs(fracpath)})

            for a in tqdm(pat_arcs, leave=False):
                testdir = os.path.join(tset.path, 'data', 'test', f'PAT{p}', f'Fx{f:02d}', f'Arc{a:02d}')
                preddir = os.path.join(tset.path, 'data', 'predictions', model_name, f'epoch_{epoch}', f'PAT{p}', f'Fx{f:02d}', f'Arc{a:02d}')
                os.makedirs(preddir, exist_ok=True)

                image_files = sorted(fn for fn in os.listdir(testdir) if fn.endswith('_image.npy'))

                for fname in tqdm(image_files, desc='Predicting', leave=False):
                    image = np.load(os.path.join(testdir, fname))
                    label = np.load(os.path.join(testdir, fname.replace('_image.npy', '_label.npy')))
                    x = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)

                    with torch.no_grad():
                        pred = model.netG(x)

                    pred = pred.squeeze().cpu().numpy()

                    # Threshold the predictions.
                    pred = np.where(pred > 0.5, 1.0, 0.0)
                    pred = pred.astype(np.bool_)

                    basename = fname.replace('_image.npy', '')
                    savepath = os.path.join(preddir, f'{basename}_pred.npy')
                    print(savepath)
                    np.save(savepath, pred)

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(image.T, cmap='gray', origin='lower')
                    axes[0].set_title('Input')
                    axes[0].axis('off')
                    axes[1].imshow(label.T, cmap='gray', vmin=0, vmax=1, origin='lower')
                    axes[1].set_title('Ground Truth (GTV)')
                    axes[1].axis('off')
                    axes[2].imshow(pred.T, cmap='gray', vmin=0, vmax=1, origin='lower')
                    axes[2].set_title('Prediction (GTV)')
                    axes[2].axis('off')
                    fig.tight_layout()
                    fig.savefig(os.path.join(preddir, f'{basename}_vis.png'), dpi=150, bbox_inches='tight')
                    plt.close(fig)
