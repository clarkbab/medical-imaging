from mymi.evaluation.dataset.nifti import create_multi_segmenter_evaluation

dataset = ['PMCC-HN-TEST-112', 'PMCC-HN-TRAIN-112']
region = 
test_fold = 0

create_multi_segmenter_evaluation(dataset, region, model, n_folds=n_folds, test_fold=test_fold)


    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
