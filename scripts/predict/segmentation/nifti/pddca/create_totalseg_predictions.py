from mymi.predictions.nifti import create_totalseg_predictions

dataset = 'PDDCA'
kwargs = dict(
    dry_run=False,
    overwrite_labels=True,
    rename_regions={
        'brainstem': 'ts_Brainstem',
        'mandible': 'ts_Bone_Mandible',
        'optic_nerve_left': 'ts_OpticNrv_L',
        'optic_nerve_right': 'ts_OpticNrv_R',
        'parotid_gland_left': 'ts_Parotid_L',
        'parotid_gland_right': 'ts_Parotid_R',
        'submandibular_gland_left': 'ts_Glnd_Submand_L',
        'submandibular_gland_right': 'ts_Glnd_Submand_R',
    },
    save_as_labels=True,
    task_regions={
        'brain_structures': [
            'brainstem',
        ],
        'craniofacial_structures': [
            'mandible',
        ],
        'head_glands_cavities': [
            'optic_nerve_left',
            'optic_nerve_right',
            'parotid_gland_left',
            'parotid_gland_right',
            'submandibular_gland_left',
            'submandibular_gland_right',
        ],
    },
)
create_totalseg_predictions(dataset, **kwargs)
