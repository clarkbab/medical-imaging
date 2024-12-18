import os
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

n_folds = 5
folds = list(range(n_folds))
folds = [0]
n_regions = 17
regions = list(range(n_regions))
regions = [0]
mf_config = "/home/baclark/.mediaflux/mflux.cfg"
mf_project = "/projects/proj-4000_punim1413-1128.4.825"
raw_path = "/data/gpfs/projects/punim1413/mymi/datasets/nnunet/raw"

# for f in folds:
#     for r in regions:
#         dataset = 21 + (f * n_regions) + r
#         dataset = f"Dataset{dataset:03d}_REF_MODEL_SINGLE_REGION_{REGIONS[r]}_FOLD_{f}"
#         src_path = os.path.join(raw_path, dataset)
#         # 'dest_path' should be the parent directory to move the dataset into.
#         dest_path = os.path.join(mf_project, 'nnunet', 'raw')
#         command = [
#             'unimelb-mf-upload', 
#             '--mf.config', mf_config,
#             '--nb-workers', '4',
#             '--dest', dest_path,
#             src_path,
#         ]
#         print(command)
#         if not dry_run:
#             subprocess.run(command)

src_path = raw_path
# Dest_path is the parent directory, will create 'raw' folder under 'nnunet' folder.
dest_path = os.path.join(mf_project, 'nnunet')
command = [
    'unimelb-mf-upload', 
    '--mf.config', mf_config,
    '--nb-workers', '4',
    '--dest', dest_path,
    src_path,
]
print(command)
if not dry_run:
    subprocess.run(command)
