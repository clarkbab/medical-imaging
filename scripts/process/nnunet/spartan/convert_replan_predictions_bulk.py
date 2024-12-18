import subprocess

dry_run = False

REGIONS = [
    'Bone_Mandible',     # 0
    'BrachialPlex_L',    # 1
    'BrachialPlex_R',    # 2
    'Brain',             # 3
    'Brainstem',         # 4
    'Cavity_Oral',       # 5
    'Esophagus_S',       # 6
    'GTVp',              # 7
    'Glnd_Submand_L',    # 8
    'Glnd_Submand_R',    # 9
    'Larynx',            # 10
    'Lens_L',            # 11
    'Lens_R',            # 12
    'Musc_Constrict',    # 13
    'Parotid_L',         # 14
    'Parotid_R',         # 15
    'SpinalCord'         # 16
]

script = 'scripts/process/nnunet/spartan/convert_replan_predictions.slurm'
n_folds = 5
folds = list(range(n_folds))
folds = [0, 1]
n_regions = 17
regions = list(range(n_regions))
regions = [12]

for f in folds:
    for r in regions:
        dataset = 21 + (f * n_regions) + r
        rname = REGIONS[r]
        export = f'ALL,FOLD={f},NNUNET_DATASET={dataset:03d},REGION={rname}'
        command = [
            'sbatch', 
            '--export', export,
            script
        ]
        print(command)
        if not dry_run:
            subprocess.run(command)
