#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 120G
#SBATCH --time 0-00:30:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="HNPCT"
PAT_ID=""

def create_localiser_prediction(
    dataset: Union[str, List[str]],
    pat_id: Union[PatientID, List[PatientID]],
    localiser: Union[ModelName, Model],
    check_epochs: bool = True,
    device: Optional[torch.device] = None,
    savepath: Optional[str] = None) -> None: