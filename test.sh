datasets="['PMCC-HN-TEST-LOC','PMCC-HN-TRAIN-LOC']"
region='Brain'
test_fold=3

n_train_max=$(python scripts/utilities/get_n_train_max.py --datasets $datasets --region $region --test_fold $test_fold)
echo $n_train_max

