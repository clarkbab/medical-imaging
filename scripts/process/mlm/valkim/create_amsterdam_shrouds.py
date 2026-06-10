import os
import subprocess

from dicomset.utils import logger
import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm

from mymi.utils.cdog import list_tiff_fractions, list_tiff_arcs, get_kv_path

pat_ids = [1]
pathpath = r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files"
fractions = 'all'
fractions = [1]
arcs = 'all'
arcs = [1]

for pat_id in tqdm(pat_ids, desc='Patients'):
    pat_path = os.path.join(pathpath, f'Patient{pat_id:02d}')
    frac_ids = list_tiff_fractions(pat_path) if fractions == 'all' else fractions
    for frac_id in tqdm(frac_ids, desc='Fractions', leave=False):
        frac_path = os.path.join(pat_path, f'Fx{frac_id:02d}')
        arc_ids = list_tiff_arcs(frac_path) if arcs == 'all' else arcs
        kv_path = get_kv_path(frac_path)
        as_path = kv_path + '_AS'
        for arc_id in tqdm(arc_ids, desc='Arcs', leave=False):
            arc_dir = os.path.join(as_path, f'arc_{arc_id}')
            output_path = os.path.join(arc_dir, f'shroud.mha')
            cmd = ['rtkamsterdamshroud', '-v', '-p', arc_dir, '-r', r'frame_[0-9]+\.mha', '-o', output_path]
            logger.info(cmd)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            print(f'Arc {arc_id} shroud exit code: {result.returncode}')

            if result.returncode == 0:
                # Pad shroud to the next size whose prime factors are only 2, 3 and 5
                # (required by VNL FFT in rtkextractshroudsignal).
                def is_valid_fft_size(n):
                    for p in (2, 3, 5):
                        while n % p == 0:
                            n //= p
                    return n == 1
                shroud_img = sitk.ReadImage(output_path)
                n_frames = shroud_img.GetSize()[1]  # ITK size: (n_si, n_frames)
                pad_frames = next(n for n in range(n_frames, n_frames * 2) if is_valid_fft_size(n))
                if pad_frames > n_frames:
                    arr = sitk.GetArrayFromImage(shroud_img)  # (n_frames, n_si)
                    padded = np.pad(arr, ((0, pad_frames - n_frames), (0, 0)), mode='edge')
                    shroud_signal_input = os.path.join(arc_dir, 'shroud_padded.mha')
                    sitk.WriteImage(sitk.GetImageFromArray(padded), shroud_signal_input)
                    print(f'Arc {arc_id} shroud padded from {n_frames} to {pad_frames} frames for FFT.')
                else:
                    shroud_signal_input = output_path
                signal_path = os.path.join(arc_dir, 'signal.txt')
                phase_path = os.path.join(arc_dir, 'phase.txt')
                cmd2 = ['rtkextractshroudsignal', '-v', '-i', shroud_signal_input, '-o', signal_path, '-p', phase_path, '--model', 'LOCAL_PHASE']
                logger.info(cmd2)
                result2 = subprocess.run(cmd2, capture_output=True, text=True)
                if result2.stdout:
                    print(result2.stdout)
                if result2.stderr:
                    print(result2.stderr)
                print(f'Arc {arc_id} signal exit code: {result2.returncode}')

                if result2.returncode == 0 and pad_frames > n_frames:
                    # Crop signal and phase back to original n_frames.
                    for fpath in (signal_path, phase_path):
                        with open(fpath) as f:
                            lines = f.readlines()
                        with open(fpath, 'w') as f:
                            f.writelines(lines[:n_frames])

print('Done.')
