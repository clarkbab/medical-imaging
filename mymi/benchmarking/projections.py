from augmed.utils import arg_to_list
import numpy as np
import pandas as pd
import time
import torch
from typing import *
from tqdm import tqdm

from mymi import logging
from mymi.typing import *
from mymi.processing import create_ctorch_projections, create_diffdrr_projections, create_igt_projections

def benchmark_projections(
    volume: Volume,
    affine: Affine,
    treatment_iso: Point3D,
    sid: float,
    sdd: float,
    det_size: Size2D,
    det_spacing: Spacing2D,
    det_offset: Point2D,
    kv_source_angles: float | List[float],
    labels: LabelVolumeBatch | None = None,
    methods: Literal['igt', 'diffdrr', 'ctorch'] | List[Literal['igt', 'diffdrr', 'ctorch']] | None = None,
    n_warmup: int = 1,
    n_iter: int = 5,
    igt_dirpath: DirPath | None = None,
    diffdrr_patch_size: int = 128,
) -> pd.DataFrame:
    logging.log_args()
    if methods is None:
        methods = ['igt', 'diffdrr', 'ctorch']
    else:
        methods = arg_to_list(methods, str)

    method_fns = {
        'igt': lambda: create_igt_projections(
            volume,
            affine,
            treatment_iso,
            sid,
            sdd,
            det_size,
            det_spacing,
            det_offset,
            kv_source_angles,
            dirpath=igt_dirpath,
            labels=labels,
            recreate=True,
        ),
        'diffdrr': lambda: create_diffdrr_projections(
            volume,
            affine,
            treatment_iso,
            sid,
            sdd,
            det_size,
            det_spacing,
            det_offset,
            kv_source_angles,
            labels=labels,
            patch_size=diffdrr_patch_size,
        ),
        'ctorch': lambda: create_ctorch_projections(
            volume,
            affine,
            treatment_iso,
            sid,
            sdd,
            det_size,
            det_spacing,
            det_offset,
            kv_source_angles,
            labels=labels,
        ),
    }

    rows = []

    for method in tqdm(methods, desc='Benchmarking projection methods'):
        if method not in method_fns:
            raise ValueError(f"Unknown method '{method}'. Expected one of: 'igt', 'diffdrr', 'ctorch'.")

        fn = method_fns[method]
        logging.info(f"Benchmarking '{method}': {n_warmup} warmup + {n_iter} timed iterations.")

        # Warmup.
        for _ in tqdm(range(n_warmup), desc=f"  Warmup ({method})", leave=False):
            fn()
            torch.cuda.empty_cache()

        # Timed iterations.
        for i in tqdm(range(n_iter), desc=f"  Timed iterations ({method})", leave=False):
            start = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - start
            torch.cuda.empty_cache()
            rows.append({
                'method': method,
                'iteration': i,
                'time': elapsed,
            })

        iter_times = [r['time'] for r in rows if r['method'] == method]
        mean_time = float(np.mean(iter_times))
        std_time = float(np.std(iter_times))
        logging.info(f"  '{method}': {mean_time:.4f} ± {std_time:.4f}s per iteration.")

    df = pd.DataFrame(rows)

    # Print summary.
    logging.info("Benchmark summary:")
    for method in methods:
        method_df = df[df['method'] == method]
        mean_time = method_df['time'].mean()
        std_time = method_df['time'].std()
        logging.info(f"  {method}: {mean_time:.4f} ± {std_time:.4f}s")

    return df
