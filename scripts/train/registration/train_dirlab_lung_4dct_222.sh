DATASET="DIRLAB-LUNG-4DCT-REG-222"
PROJECT="IMREG"
MODEL="test"
N_EPOCHS=100
LR_INIT=1e-3
LOSS_FN="ncc"

command="python train_registration.py $DATASET $PROJECT $MODEL $N_EPOCHS $LR_INIT\
    --loss_fn $LOSS_FN"
echo $command
$command
