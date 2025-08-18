from mymi.predictions.nifti import create_totalseg_predictions

dataset = 'DIRLAB-LUNG-COPD'
kwargs = dict(
    combine_regions={
        'lung': 'Lung'
    },
    # remove_task_regions={
    #     'total': 'all',
    #     'other': ['some', 'regions']
    # }
    remove_task_regions='all',
    save_as_labels=True,
    task_regions={
        'total': [
            'lung_lower_lobe_left',
            'lung_lower_lobe_right',
            'lung_middle_lobe_right',
            'lung_upper_lobe_left',
            'lung_upper_lobe_right',
        ],
    }
)
create_totalseg_predictions(dataset, **kwargs)
