from dicomset.utils import resolve_filepath
import os
import subprocess
import sys
from tqdm.auto import tqdm

from mymi.utils.varian import load_xim_arc

angular_ranges = [1, 2, 5, 10, 20, 45, 90, 180, 360]

frame = 228

f = r"L:\LEARN\Orange\Lung\Patient Images\PAT01\Fx2\CBCT\CBCT1\CBCT Projections\IPS\Proj 0"
data, info = load_xim_arc(f)
angles = info['kv-source-angle'].values

c = angles[frame]

base_script_f = resolve_filepath(f'files:lar/fdk/scripts')

for r in tqdm(angular_ranges, leave=False):
    # Run script by name.
    script = f"script_{frame}_{str(c).replace('.', '_')}_{r}"
    print(script)
    result = subprocess.run([sys.executable, os.path.join(base_script_f, f"{script}.py")])
    if result.returncode != 0:
        print(f"Script '{script}' failed with exit code {result.returncode}.")
