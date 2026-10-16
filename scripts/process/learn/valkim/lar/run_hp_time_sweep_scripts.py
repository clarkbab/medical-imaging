from dicomset.utils import resolve_filepath
import os
import subprocess
import sys
from tqdm.auto import tqdm

angular_ranges = [1, 2, 5, 10, 20, 45, 90, 180, 360]
n_iters = [1, 2, 3, 5, 10]
n_cg_iters = [1, 2, 3, 5]

frame = 228         # Frame number.
c = 0.023           # Central angle.

base_script_f = resolve_filepath(f'files:lar/hp_time_sweep/scripts')

for r in tqdm(angular_ranges, leave=False):
    for n in tqdm(n_iters):
        for n_cg in tqdm(n_cg_iters, leave=False):
            # Run script by name.
            script = f"script_{n}_{n_cg}_{frame}_{str(c).replace('.', '_')}_{r}"
            print(script)
            result = subprocess.run([sys.executable, os.path.join(base_script_f, f"{script}.py")])
            if result.returncode != 0:
                print(f"Script '{script}' failed with exit code {result.returncode}.")
