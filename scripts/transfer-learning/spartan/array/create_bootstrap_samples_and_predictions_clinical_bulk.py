import subprocess

regions = '0-16'
regions = '4,5,12,13'
script = 'scripts/transfer-learning/spartan/array/create_bootstrap_samples_and_predictions_clinical.slurm'
model_types = "\"['clinical-v2']\""
metrics = "\"['apl-mm-tol-{tol}','dice','dm-surface-dice-tol-{tol}','hd','hd-95','msd']\""
stats = "\"['mean','q1','q3']\""

# Create slurm command.
export = f'ALL,MODEL_TYPES={model_types},METRICS={metrics},STATS={stats}'
command = f'sbatch --array={regions} --export={export} {script}' 
print(command)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
process.communicate()
