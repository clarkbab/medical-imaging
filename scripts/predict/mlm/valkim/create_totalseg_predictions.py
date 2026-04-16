from mymi.predictions.nifti.segmentation import create_totalseg_predictions

dataset = 'VALKIM-PP'
pats = ['PAT1', 'PAT2', 'PAT3']
# pats = ['PAT3']
# Fill in both inhale/exhale series.
# series = ['series_0', 'series_5']
# series = ['series_1', 'series_2', 'series_3', 'series_4']
series = ['series_6', 'series_7', 'series_8', 'series_9']

for p in pats:
    for s in series:
        kwargs = dict(
            dry_run=False,
            overwrite_labels=True,
            pat=p,
            # ct_series='series_0',   # Inhale.
            # regions_series='series_0',
            # ct_series='series_5',   # Exhale.
            # regions_series='series_5',
            ct_series=s,
            regions_series=s,
            rename_regions=lambda r: f"ts_{r}",
            save_as_labels=True,
            task_regions={
                # 'brain_structures': [
                #     'brainstem',
                # ],
                # 'craniofacial_structures': [
                #     'mandible',
                # ],
                # 'head_glands_cavities': [
                #     'optic_nerve_left',
                #     'optic_nerve_right',
                #     'parotid_gland_left',
                #     'parotid_gland_right',
                #     'submandibular_gland_left',
                #     'submandibular_gland_right',
                # ],
                'total': [
                    'clavicula_left',
                    'clavicula_right',
                    'costal_cartilages',
                    'esophagus',
                    'heart',
                    'lung_upper_lobe_left',
                    'lung_lower_lobe_left',
                    'lung_upper_lobe_right',
                    'lung_middle_lobe_right',
                    'lung_lower_lobe_right',
                    *[f'rib_left_{i}' for i in range(1, 12)],
                    *[f'rib_right_{i}' for i in range(1, 12)],
                    'scapula_left',
                    'scapula_right',
                    'spinal_cord',
                    'sternum',
                    'trachea',
                    *[f'vertebrae_C{i}' for i in range(1, 8)],
                    *[f'vertebrae_L{i}' for i in range(1, 5)],
                    'vertebrae_S1',
                    *[f'vertebrae_T{i}' for i in range(1, 12)],
                ],
            },
        )
        create_totalseg_predictions(dataset, **kwargs)
