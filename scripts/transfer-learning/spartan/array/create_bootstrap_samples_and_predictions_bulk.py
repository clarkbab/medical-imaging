import subprocess

regions = '0-16'
regions = '1-16'
script = 'scripts/transfer-learning/spartan/array/create_bootstrap_samples_and_predictions.slurm'
model_types = "\"['clinical','transfer']\""
metrics = "\"['apl-mm-tol-{tol}','dice','dm-surface-dice-tol-{tol}','hd','hd-95','msd']\""
stats = "\"['mean','q1','q3']\""

# Create slurm command.
export = f'ALL,MODEL_TYPES={model_types},METRICS={metrics},STATS={stats}'
command = f'sbatch --array={regions} --export={export} {script}' 
print(command)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
process.communicate()
