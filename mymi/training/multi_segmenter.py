from datetime import datetime
import json
import numpy as np
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner
from typing import List, Optional, Union

from mymi import config
from mymi.loaders import MultiLoader
from mymi.loaders.augmentation import get_transforms
from mymi.loaders.hooks import naive_crop
from mymi import logging
from mymi.losses import DiceLoss, DiceWithFocalLoss
from mymi.models.systems import MultiSegmenter, Segmenter
from mymi.reporting.loaders import get_multi_loader_manifest
from mymi.utils import arg_to_list

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_multi_segmenter(
    dataset: Union[str, List[str]],
    region: str,
    model_name: str,
    run_name: str,
    batch_size: int = 1,
    ckpt_model: bool = True,
    dynamic_weights_factor: float = 1,
    dynamic_weights_convergence_delay: int = 20,
    dynamic_weights_convergence_delay_2: int = 5,
    dynamic_weights_convergence_thresholds: List[float] = [],
    halve_channels: bool = False,
    include_background: bool = False,
    lam: float = 0.5,
    loss_fn: str = 'dice_with_focal',
    lr_find: bool = False,
    lr_find_iter: int = 1e3,
    lr_find_min_lr: float = 1e-6,
    lr_find_max_lr: float = 1e3,
    lr_init: float = 1e-3,
    lr_milestones: List[int] = [],
    n_epochs: int = 100,
    n_folds: Optional[int] = None,
    n_gpus: int = 1,
    n_nodes: int = 1,
    n_workers: int = 1,
    n_split_channels: int = 2,
    p_val: float = 0.2,
    precision: Union[str, int] = 'bf16',
    random_seed: float = 42,
    resume: bool = False,
    resume_run: Optional[str] = None,
    resume_ckpt: str = 'last',
    slurm_job_id: Optional[str] = None,
    slurm_array_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    stand_mean: Optional[float] = None,
    stand_std: Optional[float] = None,
    test_fold: Optional[int] = None,
    thresh_low: Optional[float] = None,
    thresh_high: Optional[float] = None,
    use_augmentation: bool = True,
    use_dynamic_weights: bool = False,
    use_loader_split_file: bool = False,
    use_logger: bool = False,
    use_lr_scheduler: bool = False,
    use_stand: bool = False,
    use_thresh: bool = False,
    use_weighting: bool = False,
    weight_decay: float = 0,
    weighting_scheme: Optional[int] = None) -> None:
    logging.arg_log('Training model', ('dataset', 'region', 'model_name', 'run_name'), (dataset, region, model_name, run_name))
    regions = arg_to_list(region, str)

    # Allow for reproducible training runs.
    seed_everything(random_seed, workers=True)

    # Get augmentation transforms.
    if use_augmentation:
        transform_train, transform_val = get_transforms(thresh_high=thresh_high, thresh_low=thresh_low, use_stand=use_stand, use_thresh=use_thresh)
    else:
        transform_train = None
        transform_val = None
    logging.info(f"Training transform: {transform_train}")
    logging.info(f"Validation transform: {transform_val}")

    # Define loss function.
    if loss_fn == 'dice':
        loss_fn = DiceLoss()
    elif loss_fn == 'dice_with_focal':
        loss_fn = DiceWithFocalLoss(lam=lam)

    # Create data loaders.
    train_loader, val_loader, _ = MultiLoader.build_loaders(dataset, batch_size=batch_size, data_hook=naive_crop, include_background=include_background, n_folds=n_folds, n_workers=n_workers, p_val=p_val, region=regions, test_fold=test_fold, transform_train=transform_train, transform_val=transform_val, use_split_file=use_loader_split_file)

    # Create weighting scheme.
    if use_weighting:
        if weighting_scheme == '1a':
            # Default (frequency) weighting.
            weights = None
            weights_schedule = None
        elif weighting_scheme == '2a':
            # Volume weighting for all epochs.
            weights = [[
                0,
                0.0034936,
                0.00722492,
                0.02615894,
                0.02589116,
                0.33027497,
                0.2783234,
                0.31547564,
                0.00664951,
                0.00650785
            ]]
            weights_schedule = [0]
        elif weighting_scheme == '2b':
            # Volume weighting, and back to frequency after 2k epochs.
            weights_1 = [
                0,
                0.0034936,
                0.00722492,
                0.02615894,
                0.02589116,
                0.33027497,
                0.2783234,
                0.31547564,
                0.00664951,
                0.00650785
            ]
            weights = [
                weights_1,
                None
            ]
            weights_schedule = [0, 2000]
        elif weighting_scheme == '2c':
            # Volume weighting, and back to frequency after 1k epochs.
            weights_1 = [
                0,
                0.0034936,
                0.00722492,
                0.02615894,
                0.02589116,
                0.33027497,
                0.2783234,
                0.31547564,
                0.00664951,
                0.00650785
            ]
            weights = [
                weights_1,
                None
            ]
            weights_schedule = [0, 1000]
        elif weighting_scheme == '3a':
            # Bring OARs into play in groups based on volume.
            Bone_Mandible_idx = 1
            Brainstem_idx = 2
            Glnd_Submand_L_idx = 3
            Glnd_Submand_R_idx = 4
            OpticChiasm_idx = 5
            OpticNrv_L_idx = 6
            OpticNrv_R_idx = 7
            Parotid_L_idx = 8
            Parotid_R_idx = 9

            # Add small OARs.
            weight_1 = [0] * (len(regions) + 1)
            weight_1[OpticChiasm_idx] = 1 / 3
            weight_1[OpticNrv_L_idx] = 1 / 3
            weight_1[OpticNrv_R_idx] = 1 / 3

            # Add medium OARs.
            weight_2 = [0] * (len(regions) + 1)
            weight_2[Glnd_Submand_L_idx] = 1 / 5
            weight_2[Glnd_Submand_R_idx] = 1 / 5
            weight_2[OpticChiasm_idx] = 1 / 5
            weight_2[OpticNrv_L_idx] = 1 / 5
            weight_2[OpticNrv_R_idx] = 1 / 5

            # Add large OARs.
            weight_3 = [0] * (len(regions) + 1)
            weight_3[Brainstem_idx] = 1 / 8
            weight_3[Glnd_Submand_L_idx] = 1 / 8
            weight_3[Glnd_Submand_R_idx] = 1 / 8
            weight_3[OpticChiasm_idx] = 1 / 8
            weight_3[OpticNrv_L_idx] = 1 / 8
            weight_3[OpticNrv_R_idx] = 1 / 8
            weight_3[Parotid_L_idx] = 1 / 8
            weight_3[Parotid_R_idx] = 1 / 8

            # Add Mandible.
            weight_4 = [0] * (len(regions) + 1)
            weight_4[Bone_Mandible_idx] = 1 / 9
            weight_4[Brainstem_idx] = 1 / 9
            weight_4[Glnd_Submand_L_idx] = 1 / 9
            weight_4[Glnd_Submand_R_idx] = 1 / 9
            weight_4[OpticChiasm_idx] = 1 / 9
            weight_4[OpticNrv_L_idx] = 1 / 9
            weight_4[OpticNrv_R_idx] = 1 / 9
            weight_4[Parotid_L_idx] = 1 / 9
            weight_4[Parotid_R_idx] = 1 / 9
            weights = [
                weight_1,
                weight_2,
                weight_3,
                weight_4,
                None
            ]
            weights_schedule = [0, 1000, 2000, 3000, 4000]
        elif weighting_scheme == '3b':
            # Bring OARs into play in groups based on volume.
            Bone_Mandible_idx = 1
            Brainstem_idx = 2
            Glnd_Submand_L_idx = 3
            Glnd_Submand_R_idx = 4
            OpticChiasm_idx = 5
            OpticNrv_L_idx = 6
            OpticNrv_R_idx = 7
            Parotid_L_idx = 8
            Parotid_R_idx = 9

            # Add small OARs.
            weight_1 = [0] * (len(regions) + 1)
            weight_1[OpticChiasm_idx] = 1 / 3
            weight_1[OpticNrv_L_idx] = 1 / 3
            weight_1[OpticNrv_R_idx] = 1 / 3

            # Add medium OARs.
            weight_2 = [0] * (len(regions) + 1)
            weight_2[Glnd_Submand_L_idx] = 1 / 5
            weight_2[Glnd_Submand_R_idx] = 1 / 5
            weight_2[OpticChiasm_idx] = 1 / 5
            weight_2[OpticNrv_L_idx] = 1 / 5
            weight_2[OpticNrv_R_idx] = 1 / 5

            # Add large OARs.
            weight_3 = [0] * (len(regions) + 1)
            weight_3[Brainstem_idx] = 1 / 8
            weight_3[Glnd_Submand_L_idx] = 1 / 8
            weight_3[Glnd_Submand_R_idx] = 1 / 8
            weight_3[OpticChiasm_idx] = 1 / 8
            weight_3[OpticNrv_L_idx] = 1 / 8
            weight_3[OpticNrv_R_idx] = 1 / 8
            weight_3[Parotid_L_idx] = 1 / 8
            weight_3[Parotid_R_idx] = 1 / 8

            # Add Mandible.
            weight_4 = [0] * (len(regions) + 1)
            weight_4[Bone_Mandible_idx] = 1 / 9
            weight_4[Brainstem_idx] = 1 / 9
            weight_4[Glnd_Submand_L_idx] = 1 / 9
            weight_4[Glnd_Submand_R_idx] = 1 / 9
            weight_4[OpticChiasm_idx] = 1 / 9
            weight_4[OpticNrv_L_idx] = 1 / 9
            weight_4[OpticNrv_R_idx] = 1 / 9
            weight_4[Parotid_L_idx] = 1 / 9
            weight_4[Parotid_R_idx] = 1 / 9
            weights = [
                weight_1,
                weight_2,
                weight_3,
                weight_4,
                None
            ]
            weights_schedule = [0, 2000, 4000, 6000, 8000]
        elif weighting_scheme == '4a':
            # Bring OARs into play in groups and focus on recent OARs.
            focus_weight = 0.6
            Bone_Mandible_idx = 1
            Brainstem_idx = 2
            Glnd_Submand_L_idx = 3
            Glnd_Submand_R_idx = 4
            OpticChiasm_idx = 5
            OpticNrv_L_idx = 6
            OpticNrv_R_idx = 7
            Parotid_L_idx = 8
            Parotid_R_idx = 9

            # Add small OARs.
            weight_1 = [0] * (len(regions) + 1)
            weight_1[OpticChiasm_idx] = 1 / 3
            weight_1[OpticNrv_L_idx] = 1 / 3
            weight_1[OpticNrv_R_idx] = 1 / 3

            # Add medium OARs.
            weight_2 = [0] * (len(regions) + 1)
            weight_2[Glnd_Submand_L_idx] = focus_weight / 2
            weight_2[Glnd_Submand_R_idx] = focus_weight / 2
            weight_2[OpticChiasm_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_L_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_R_idx] = (1 - focus_weight) / 3

            # Add large OARs.
            weight_3 = [0] * (len(regions) + 1)
            weight_3[Brainstem_idx] = focus_weight / 3
            weight_3[Parotid_L_idx] = focus_weight / 3
            weight_3[Parotid_R_idx] = focus_weight / 3
            weight_3[Glnd_Submand_L_idx] = (1 - focus_weight) / 5
            weight_3[Glnd_Submand_R_idx] = (1 - focus_weight) / 5
            weight_3[OpticChiasm_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_L_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_R_idx] = (1 - focus_weight) / 5

            # Add Mandible.
            weight_4 = [0] * (len(regions) + 1)
            weight_4[Bone_Mandible_idx] = focus_weight
            weight_4[Brainstem_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_L_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_R_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_L_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_R_idx] = (1 - focus_weight) / 8
            weight_4[OpticChiasm_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_L_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_R_idx] = (1 - focus_weight) / 8
            weights = [
                weight_1,
                weight_2,
                weight_3,
                weight_4,
                None
            ]
            weights_schedule = [0, 1000, 2000, 3000, 4000]
        elif weighting_scheme == '4a-b':
            # Bring OARs into play in groups and focus on recent OARs.
            focus_weight = 0.6
            Bone_Mandible_idx = 1
            Brainstem_idx = 2
            Glnd_Submand_L_idx = 3
            Glnd_Submand_R_idx = 4
            OpticChiasm_idx = 5
            OpticNrv_L_idx = 6
            OpticNrv_R_idx = 7
            Parotid_L_idx = 8
            Parotid_R_idx = 9

            # Add small OARs.
            weight_1 = [0] * (len(regions) + 1)
            weight_1[OpticChiasm_idx] = 1 / 3
            weight_1[OpticNrv_L_idx] = 1 / 3
            weight_1[OpticNrv_R_idx] = 1 / 3

            # Add medium OARs.
            weight_2 = [0] * (len(regions) + 1)
            weight_2[Glnd_Submand_L_idx] = focus_weight / 2
            weight_2[Glnd_Submand_R_idx] = focus_weight / 2
            weight_2[OpticChiasm_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_L_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_R_idx] = (1 - focus_weight) / 3

            # Add large OARs.
            weight_3 = [0] * (len(regions) + 1)
            weight_3[Brainstem_idx] = focus_weight / 3
            weight_3[Parotid_L_idx] = focus_weight / 3
            weight_3[Parotid_R_idx] = focus_weight / 3
            weight_3[Glnd_Submand_L_idx] = (1 - focus_weight) / 5
            weight_3[Glnd_Submand_R_idx] = (1 - focus_weight) / 5
            weight_3[OpticChiasm_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_L_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_R_idx] = (1 - focus_weight) / 5

            # Add Mandible.
            weight_4 = [0] * (len(regions) + 1)
            weight_4[Bone_Mandible_idx] = focus_weight
            weight_4[Brainstem_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_L_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_R_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_L_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_R_idx] = (1 - focus_weight) / 8
            weight_4[OpticChiasm_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_L_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_R_idx] = (1 - focus_weight) / 8
            weights = [
                weight_1,
                weight_2,
                weight_3,
                weight_4,
                None
            ]
            weights_schedule = [0, 2000, 4000, 6000, 8000]
        elif weighting_scheme == '4b':
            # Bring OARs into play in groups and focus on recent OARs.
            focus_weight = 0.8
            Bone_Mandible_idx = 1
            Brainstem_idx = 2
            Glnd_Submand_L_idx = 3
            Glnd_Submand_R_idx = 4
            OpticChiasm_idx = 5
            OpticNrv_L_idx = 6
            OpticNrv_R_idx = 7
            Parotid_L_idx = 8
            Parotid_R_idx = 9

            # Add small OARs.
            weight_1 = [0] * (len(regions) + 1)
            weight_1[OpticChiasm_idx] = 1 / 3
            weight_1[OpticNrv_L_idx] = 1 / 3
            weight_1[OpticNrv_R_idx] = 1 / 3

            # Add medium OARs.
            weight_2 = [0] * (len(regions) + 1)
            weight_2[Glnd_Submand_L_idx] = focus_weight / 2
            weight_2[Glnd_Submand_R_idx] = focus_weight / 2
            weight_2[OpticChiasm_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_L_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_R_idx] = (1 - focus_weight) / 3

            # Add large OARs.
            weight_3 = [0] * (len(regions) + 1)
            weight_3[Brainstem_idx] = focus_weight / 3
            weight_3[Parotid_L_idx] = focus_weight / 3
            weight_3[Parotid_R_idx] = focus_weight / 3
            weight_3[Glnd_Submand_L_idx] = (1 - focus_weight) / 5
            weight_3[Glnd_Submand_R_idx] = (1 - focus_weight) / 5
            weight_3[OpticChiasm_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_L_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_R_idx] = (1 - focus_weight) / 5

            # Add Mandible.
            weight_4 = [0] * (len(regions) + 1)
            weight_4[Bone_Mandible_idx] = focus_weight
            weight_4[Brainstem_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_L_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_R_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_L_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_R_idx] = (1 - focus_weight) / 8
            weight_4[OpticChiasm_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_L_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_R_idx] = (1 - focus_weight) / 8
            weights = [
                weight_1,
                weight_2,
                weight_3,
                weight_4,
                None
            ]
            weights_schedule = [0, 1000, 2000, 3000, 4000]
        elif weighting_scheme == '4b-b':
            # Bring OARs into play in groups and focus on recent OARs.
            focus_weight = 0.8
            Bone_Mandible_idx = 1
            Brainstem_idx = 2
            Glnd_Submand_L_idx = 3
            Glnd_Submand_R_idx = 4
            OpticChiasm_idx = 5
            OpticNrv_L_idx = 6
            OpticNrv_R_idx = 7
            Parotid_L_idx = 8
            Parotid_R_idx = 9

            # Add small OARs.
            weight_1 = [0] * (len(regions) + 1)
            weight_1[OpticChiasm_idx] = 1 / 3
            weight_1[OpticNrv_L_idx] = 1 / 3
            weight_1[OpticNrv_R_idx] = 1 / 3

            # Add medium OARs.
            weight_2 = [0] * (len(regions) + 1)
            weight_2[Glnd_Submand_L_idx] = focus_weight / 2
            weight_2[Glnd_Submand_R_idx] = focus_weight / 2
            weight_2[OpticChiasm_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_L_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_R_idx] = (1 - focus_weight) / 3

            # Add large OARs.
            weight_3 = [0] * (len(regions) + 1)
            weight_3[Brainstem_idx] = focus_weight / 3
            weight_3[Parotid_L_idx] = focus_weight / 3
            weight_3[Parotid_R_idx] = focus_weight / 3
            weight_3[Glnd_Submand_L_idx] = (1 - focus_weight) / 5
            weight_3[Glnd_Submand_R_idx] = (1 - focus_weight) / 5
            weight_3[OpticChiasm_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_L_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_R_idx] = (1 - focus_weight) / 5

            # Add Mandible.
            weight_4 = [0] * (len(regions) + 1)
            weight_4[Bone_Mandible_idx] = focus_weight
            weight_4[Brainstem_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_L_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_R_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_L_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_R_idx] = (1 - focus_weight) / 8
            weight_4[OpticChiasm_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_L_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_R_idx] = (1 - focus_weight) / 8
            weights = [
                weight_1,
                weight_2,
                weight_3,
                weight_4,
                None
            ]
            weights_schedule = [0, 2000, 4000, 6000, 8000]
        elif weighting_scheme == '4c':
            # Bring OARs into play in groups and focus on recent OARs.
            focus_weight = 0.9
            Bone_Mandible_idx = 1
            Brainstem_idx = 2
            Glnd_Submand_L_idx = 3
            Glnd_Submand_R_idx = 4
            OpticChiasm_idx = 5
            OpticNrv_L_idx = 6
            OpticNrv_R_idx = 7
            Parotid_L_idx = 8
            Parotid_R_idx = 9

            # Add small OARs.
            weight_1 = [0] * (len(regions) + 1)
            weight_1[OpticChiasm_idx] = 1 / 3
            weight_1[OpticNrv_L_idx] = 1 / 3
            weight_1[OpticNrv_R_idx] = 1 / 3

            # Add medium OARs.
            weight_2 = [0] * (len(regions) + 1)
            weight_2[Glnd_Submand_L_idx] = focus_weight / 2
            weight_2[Glnd_Submand_R_idx] = focus_weight / 2
            weight_2[OpticChiasm_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_L_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_R_idx] = (1 - focus_weight) / 3

            # Add large OARs.
            weight_3 = [0] * (len(regions) + 1)
            weight_3[Brainstem_idx] = focus_weight / 3
            weight_3[Parotid_L_idx] = focus_weight / 3
            weight_3[Parotid_R_idx] = focus_weight / 3
            weight_3[Glnd_Submand_L_idx] = (1 - focus_weight) / 5
            weight_3[Glnd_Submand_R_idx] = (1 - focus_weight) / 5
            weight_3[OpticChiasm_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_L_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_R_idx] = (1 - focus_weight) / 5

            # Add Mandible.
            weight_4 = [0] * (len(regions) + 1)
            weight_4[Bone_Mandible_idx] = focus_weight
            weight_4[Brainstem_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_L_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_R_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_L_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_R_idx] = (1 - focus_weight) / 8
            weight_4[OpticChiasm_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_L_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_R_idx] = (1 - focus_weight) / 8
            weights = [
                weight_1,
                weight_2,
                weight_3,
                weight_4,
                None
            ]
            weights_schedule = [0, 1000, 2000, 3000, 4000]
        elif weighting_scheme == '4c-b':
            # Bring OARs into play in groups and focus on recent OARs.
            focus_weight = 0.9
            Bone_Mandible_idx = 1
            Brainstem_idx = 2
            Glnd_Submand_L_idx = 3
            Glnd_Submand_R_idx = 4
            OpticChiasm_idx = 5
            OpticNrv_L_idx = 6
            OpticNrv_R_idx = 7
            Parotid_L_idx = 8
            Parotid_R_idx = 9

            # Add small OARs.
            weight_1 = [0] * (len(regions) + 1)
            weight_1[OpticChiasm_idx] = 1 / 3
            weight_1[OpticNrv_L_idx] = 1 / 3
            weight_1[OpticNrv_R_idx] = 1 / 3

            # Add medium OARs.
            weight_2 = [0] * (len(regions) + 1)
            weight_2[Glnd_Submand_L_idx] = focus_weight / 2
            weight_2[Glnd_Submand_R_idx] = focus_weight / 2
            weight_2[OpticChiasm_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_L_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_R_idx] = (1 - focus_weight) / 3

            # Add large OARs.
            weight_3 = [0] * (len(regions) + 1)
            weight_3[Brainstem_idx] = focus_weight / 3
            weight_3[Parotid_L_idx] = focus_weight / 3
            weight_3[Parotid_R_idx] = focus_weight / 3
            weight_3[Glnd_Submand_L_idx] = (1 - focus_weight) / 5
            weight_3[Glnd_Submand_R_idx] = (1 - focus_weight) / 5
            weight_3[OpticChiasm_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_L_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_R_idx] = (1 - focus_weight) / 5

            # Add Mandible.
            weight_4 = [0] * (len(regions) + 1)
            weight_4[Bone_Mandible_idx] = focus_weight
            weight_4[Brainstem_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_L_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_R_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_L_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_R_idx] = (1 - focus_weight) / 8
            weight_4[OpticChiasm_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_L_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_R_idx] = (1 - focus_weight) / 8
            weights = [
                weight_1,
                weight_2,
                weight_3,
                weight_4,
                None
            ]
            weights_schedule = [0, 2000, 4000, 6000, 8000]
        elif weighting_scheme == '4d':
            # Bring OARs into play in groups and focus on recent OARs.
            focus_weight = 0.95
            Bone_Mandible_idx = 1
            Brainstem_idx = 2
            Glnd_Submand_L_idx = 3
            Glnd_Submand_R_idx = 4
            OpticChiasm_idx = 5
            OpticNrv_L_idx = 6
            OpticNrv_R_idx = 7
            Parotid_L_idx = 8
            Parotid_R_idx = 9

            # Add small OARs.
            weight_1 = [0] * (len(regions) + 1)
            weight_1[OpticChiasm_idx] = 1 / 3
            weight_1[OpticNrv_L_idx] = 1 / 3
            weight_1[OpticNrv_R_idx] = 1 / 3

            # Add medium OARs.
            weight_2 = [0] * (len(regions) + 1)
            weight_2[Glnd_Submand_L_idx] = focus_weight / 2
            weight_2[Glnd_Submand_R_idx] = focus_weight / 2
            weight_2[OpticChiasm_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_L_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_R_idx] = (1 - focus_weight) / 3

            # Add large OARs.
            weight_3 = [0] * (len(regions) + 1)
            weight_3[Brainstem_idx] = focus_weight / 3
            weight_3[Parotid_L_idx] = focus_weight / 3
            weight_3[Parotid_R_idx] = focus_weight / 3
            weight_3[Glnd_Submand_L_idx] = (1 - focus_weight) / 5
            weight_3[Glnd_Submand_R_idx] = (1 - focus_weight) / 5
            weight_3[OpticChiasm_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_L_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_R_idx] = (1 - focus_weight) / 5

            # Add Mandible.
            weight_4 = [0] * (len(regions) + 1)
            weight_4[Bone_Mandible_idx] = focus_weight
            weight_4[Brainstem_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_L_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_R_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_L_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_R_idx] = (1 - focus_weight) / 8
            weight_4[OpticChiasm_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_L_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_R_idx] = (1 - focus_weight) / 8
            weights = [
                weight_1,
                weight_2,
                weight_3,
                weight_4,
                None
            ]
            weights_schedule = [0, 1000, 2000, 3000, 4000]
        elif weighting_scheme == '4d-b':
            # Bring OARs into play in groups and focus on recent OARs.
            focus_weight = 0.95
            Bone_Mandible_idx = 1
            Brainstem_idx = 2
            Glnd_Submand_L_idx = 3
            Glnd_Submand_R_idx = 4
            OpticChiasm_idx = 5
            OpticNrv_L_idx = 6
            OpticNrv_R_idx = 7
            Parotid_L_idx = 8
            Parotid_R_idx = 9

            # Add small OARs.
            weight_1 = [0] * (len(regions) + 1)
            weight_1[OpticChiasm_idx] = 1 / 3
            weight_1[OpticNrv_L_idx] = 1 / 3
            weight_1[OpticNrv_R_idx] = 1 / 3

            # Add medium OARs.
            weight_2 = [0] * (len(regions) + 1)
            weight_2[Glnd_Submand_L_idx] = focus_weight / 2
            weight_2[Glnd_Submand_R_idx] = focus_weight / 2
            weight_2[OpticChiasm_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_L_idx] = (1 - focus_weight) / 3
            weight_2[OpticNrv_R_idx] = (1 - focus_weight) / 3

            # Add large OARs.
            weight_3 = [0] * (len(regions) + 1)
            weight_3[Brainstem_idx] = focus_weight / 3
            weight_3[Parotid_L_idx] = focus_weight / 3
            weight_3[Parotid_R_idx] = focus_weight / 3
            weight_3[Glnd_Submand_L_idx] = (1 - focus_weight) / 5
            weight_3[Glnd_Submand_R_idx] = (1 - focus_weight) / 5
            weight_3[OpticChiasm_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_L_idx] = (1 - focus_weight) / 5
            weight_3[OpticNrv_R_idx] = (1 - focus_weight) / 5

            # Add Mandible.
            weight_4 = [0] * (len(regions) + 1)
            weight_4[Bone_Mandible_idx] = focus_weight
            weight_4[Brainstem_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_L_idx] = (1 - focus_weight) / 8
            weight_4[Parotid_R_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_L_idx] = (1 - focus_weight) / 8
            weight_4[Glnd_Submand_R_idx] = (1 - focus_weight) / 8
            weight_4[OpticChiasm_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_L_idx] = (1 - focus_weight) / 8
            weight_4[OpticNrv_R_idx] = (1 - focus_weight) / 8
            weights = [
                weight_1,
                weight_2,
                weight_3,
                weight_4,
                None
            ]
            weights_schedule = [0, 2000, 4000, 6000, 8000]
        elif weighting_scheme == '5a':
            # Only bring small OARs in (for LR find).
            OpticChiasm_idx = 5
            OpticNrv_L_idx = 6
            OpticNrv_R_idx = 7

            # Add small OARs.
            weight = [0] * (len(regions) + 1)
            weight[OpticChiasm_idx] = 1 / 3
            weight[OpticNrv_L_idx] = 1 / 3
            weight[OpticNrv_R_idx] = 1 / 3
            weights = [
                weight
            ]
            weights_schedule = [0]
        elif weighting_scheme == '5b':
            # Only bring medium OARs in (for LR find).
            Glnd_Submand_L_idx = 3
            Glnd_Submand_R_idx = 4

            # Add medium OARs.
            weight = [0] * (len(regions) + 1)
            weight[Glnd_Submand_L_idx] = 1 / 2
            weight[Glnd_Submand_R_idx] = 1 / 2
            weights = [
                weight
            ]
            weights_schedule = [0]
        elif weighting_scheme == '5c':
            # Only bring large OARs in (for LR find).
            Brainstem_idx = 2
            Parotid_L_idx = 8
            Parotid_R_idx = 9

            # Add large OARs.
            weight = [0] * (len(regions) + 1)
            weight[Brainstem_idx] = 1 / 3
            weight[Parotid_L_idx] = 1 / 3
            weight[Parotid_R_idx] = 1 / 3
            weights = [
                weight
            ]
            weights_schedule = [0]
        elif weighting_scheme == '5d':
            # Only bring extra large OAR in (for LR find).
            Bone_Mandible_idx = 1

            # Add large OARs.
            weight = [0] * (len(regions) + 1)
            weight[Bone_Mandible_idx] = 1
            weights = [
                weight
            ]
            weights_schedule = [0]
        elif weighting_scheme == '6a':
            # LR=1e=3 to 1e-4 after 2k epochs.
            use_lr_scheduler = True
            lr_milestones = [2000]
            weights = None
            weights_schedule = None
        else:
            raise ValueError(f"Invalid weighting_scheme scheme: {weighting_scheme}.")
    else:
        weights = None
        weights_schedule = None

    # Validate weights.
    if weights is not None:
        assert len(weights) == len(weights_schedule)
        for w, e in zip(weights, weights_schedule):
            if w is not None and np.sum(w).round(decimals=3) != 1:
                raise ValueError(f"Weights for epoch {e} don't sum to 1 (3dp).")

    # Create model.
    model = MultiSegmenter(
        dynamic_weights_factor=dynamic_weights_factor,
        dynamic_weights_convergence_delay=dynamic_weights_convergence_delay,
        dynamic_weights_convergence_delay_2=dynamic_weights_convergence_delay_2,
        dynamic_weights_convergence_thresholds=dynamic_weights_convergence_thresholds,
        loss=loss_fn,
        lr_init=lr_init,
        lr_milestones=lr_milestones,
        halve_channels=halve_channels,
        metrics=['dice'],
        n_gpus=n_gpus,
        n_split_channels=n_split_channels,
        region=regions,
        use_dynamic_weights=use_dynamic_weights,
        use_lr_scheduler=use_lr_scheduler,
        weights=weights,
        weight_decay=weight_decay,
        weights_schedule=weights_schedule)

    # Create logger.
    if use_logger:
        logger = WandbLogger(
            # group=f"{model_name}-{run_name}",
            project=model_name,
            name=run_name,
            save_dir=config.directories.reports)
        logger.watch(model, log='all') # Caused multi-GPU training to hang.
    else:
        logger = False

    # Create callbacks.
    ckpts_path = os.path.join(config.directories.models, model_name, run_name)
    callbacks = [
        # EarlyStopping(
        #     monitor='val/loss',
        #     patience=5),
    ]
    if ckpt_model:
        callbacks.append(ModelCheckpoint(
            auto_insert_metric_name=False,
            dirpath=ckpts_path,
            filename='loss={val/loss:.6f}-epoch={epoch}-step={trainer/global_step}',
            every_n_epochs=1,
            monitor='val/loss',
            save_last=True,
            save_top_k=1))
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    # Add optional trainer args.
    opt_kwargs = {}
    if resume:
        if resume_run is not None:
            ckpt_path = os.path.join(config.directories.models, model_name, resume_run, f'{resume_ckpt}.ckpt')
        else:
            ckpt_path = os.path.join(ckpts_path, f'{resume_ckpt}.ckpt')
        opt_kwargs['ckpt_path'] = ckpt_path
    
    # Perform training.
    trainer = Trainer(
        accelerator='gpu' if n_gpus > 0 else 'cpu',
        callbacks=callbacks,
        devices=list(range(n_gpus)) if n_gpus > 0 else 1,
        logger=logger,
        max_epochs=n_epochs,
        num_nodes=n_nodes,
        num_sanity_val_steps=2,
        precision=precision)

    if lr_find:
        tuner = Tuner(trainer)
        lr = tuner.lr_find(model, train_loader, val_loader, early_stop_threshold=None, min_lr=lr_find_min_lr, max_lr=lr_find_max_lr, num_training=lr_find_iter)
        logging.info(lr.results)
        filepath = os.path.join(config.directories.models, model_name, run_name, 'lr-finder.json')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(json.dumps(lr.results))

        # Don't proceed with training.
        return

    # Save training information.
    man_df = get_multi_loader_manifest(dataset, n_folds=n_folds, region=regions, test_fold=test_fold, use_split_file=use_loader_split_file)
    folderpath = os.path.join(config.directories.runs, model_name, run_name, datetime.now().strftime(DATETIME_FORMAT))
    os.makedirs(folderpath, exist_ok=True)
    filepath = os.path.join(folderpath, 'multi-loader-manifest.csv')
    man_df.to_csv(filepath, index=False)

    # Train the model.
    trainer.fit(model, train_loader, val_loader, **opt_kwargs)
