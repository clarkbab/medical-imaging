from mymi.processing.nifti import create_pddca_cropped_dataset

kwargs = dict(
    dry_run=False,
    recreate=True,
)
create_pddca_cropped_dataset(**kwargs)
