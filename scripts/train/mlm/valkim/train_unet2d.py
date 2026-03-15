from mymi.training.mlm.valkim import train_segmentation

dataset = 'VALKIM-PP'
pat = 'PAT1'
project = 'MLM-VALKIM'
arch = 'unet2d:m'
model = f"{arch}-{pat}-001"

train_segmentation(
    dataset,
    pat,
    project,
    model,
    n_epochs=100,
    lr_init=1e-3,
    arch=arch,
    batch_size=32,
    loss_fn='dice',
    resume=False,
)
