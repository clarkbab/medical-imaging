from multiprocessing import freeze_support

from mymi.training.mlm.valkim import train_segmentation

if __name__ == '__main__':
    freeze_support()

    dataset = 'VALKIM-PP'
    pat = 'PAT1'
    project = 'MLM-VALKIM'
    arch = 'unet2d:m'
    model = f"{arch.replace(':', '_')}-{pat}-001"

    train_segmentation(
        dataset,
        pat,
        project,
        model,
        n_epochs=100,
        n_train_angles=100,
        n_val_angles=10,
        n_val_volumes=3,
        lr_find=True,
        lr_init=1e-3,
        arch=arch,
        batch_size=16,
        loss_fn='dice',
        resume=False,
        use_logging=True,
    )
