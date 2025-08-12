import argparse
from datetime import timedelta
import os
import re
from typing import *

from mymi import logging
from mymi.typing import *

from .utils import arg_to_list

CPU_PARTITIONS =[
    'cascade',
    'sapphire'
]
GPU_PARTITIONS = [
    'gpu-h100',
    'gpu-a100',
    'gpu-a100-short',
    'feit-gpu-a100',
]
DEFAULT_CPU = 'sapphire'
DEFAULT_GPU = 'gpu-h100'

def create_slurm(
    file: str,
    array: Optional[str] = None,
    enqueue: bool = True,
    memory: int = 128,
    mode: Literal['cpu', 'gpu'] = 'gpu',
    partitions: Optional[str] = None,
    suffix: str = '',
    time: timedelta = 'days:7',
    **kwargs) -> None:
    # Handle arguments.
    if partitions is None:
        # Set the partitions based on mode.
        partitions = [DEFAULT_CPU] if mode == 'cpu' else [DEFAULT_GPU]
    elif partitions == 'go-fishing':
        partitions = GPU_PARTITIONS
    else:
        partitions = partitions.split(',')  # Fire won't interpret as a list.

    # Check partitions.
    for p in partitions:
        if p not in CPU_PARTITIONS + GPU_PARTITIONS:
            raise ValueError(f"Partition {p} not recognised. Must be one of {CPU_PARTITIONS + GPU_PARTITIONS}.")

    # Set the mode based on requested partitions.
    p0 = partitions[0]
    if p0 in CPU_PARTITIONS:
        mode = 'cpu'
    elif p0 in GPU_PARTITIONS:
        mode = 'gpu' 

    # Check consistent mode.
    for p in partitions:
        if (mode == 'cpu' and p not in CPU_PARTITIONS) or (mode == 'gpu' and p not in GPU_PARTITIONS):
            raise ValueError(f"Partitions should be either CPU or GPU, not both.")

    # Parse time string.
    if isinstance(time, str):
        match = re.match(r'(days:(\d+))?:?(hours:(\d+))?', time)
        days = int(match.group(2)) if match.group(2) is not None else 0
        hours = int(match.group(4)) if match.group(4) is not None else 0
        time = timedelta(days=days, hours=hours)

    for p in partitions:
        # Handle partition specifics.
        qos = ''
        p_memory = memory
        p_time = time
        if p == 'feit-gpu-a100':
            qos = """
#SBATCH --qos feit"""
        elif p == 'gpu-a100-short':
            max_memory = 120
            if p_memory > max_memory:
                print(f"Partition memory limit exceeded. Setting memory to {max_memory}GB.")
                p_memory = max_memory
            max_time = timedelta(hours=4)
            if p_time > max_time:
                print(f"Partition time limit exceeded. Setting time to {max_time}.")
                p_time = max_time

        # Convert time.
        days = p_time.days
        hours, remainder = divmod(p_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Convert kwargs.
        kwarg_str = ''
        for k, v in kwargs.items():
            kwarg_str += f'--{k} "{v}" '

        # Manage resources.
        g_res = """
#SBATCH --gres gpu:1""" if mode == 'gpu' else ''
        c_res = """
#SBATCH --cpus-per-gpu 8""" if mode == 'gpu' else ''

        # Create content.
        content = f"""\
#!/bin/bash
#SBATCH --account punim1413{qos}
#SBATCH --partition {p}
#SBATCH --nodes 1
#SBATCH --mem {p_memory}G{g_res}{c_res}
#SBATCH --time {days}-{hours:02}:{minutes:02}:{seconds:02}

# Host not set for slurm jobs.
export HOST="spartan"
export DOTFILES_HOME="$HOME/code/dotfiles"
source $DOTFILES_HOME/.profile

version=$(python --version)
echo $version

python {file} {kwarg_str}
"""

        # Write to file.
        p_file = file.replace('.py', '')
        suffix = f'-{suffix}' if suffix != '' else ''
        filepath = f'{p_file}-{p}{suffix}.slurm'
        print(f"Writing to {filepath}.")
        with open(filepath, 'w') as f:
            f.write(content)

        # Queue job.
        if enqueue:
            command = f'sbatch '
            if array is not None:
                command += f'--array={array} '
            command += f'{filepath}'

            print(f"Queueing job with command: {command}")
            os.system(command)

def create_slurm_grid(
    create_slurm_fn,
    *args,
    dry_run: bool = True,
    params: Union[str, Sequence[str]] = [],     # M params.
    values: Union[float, int, str, List[Union[float, int, str]], List[List[Union[float, int, str]]]] = [],     # N runs.
    **kwargs) -> None:
    # Shouldn't this work differently, it's not really a grid search? I.e. product of a,b values?
    # E.g. --params a,b --values "[[1, 2],[3, 4]]" should produce four runs:
    # a=1, b=3
    # a=1, b=4
    # a=2, b=3
    # a=2, b=4
    # Not two runs
    # a=1, b=3
    # a=2, b=4
    # Variations:
    # --params a --values 1                     N x M = 1 x 1
    # --params a --values 1,2                   N x M = 2 x 1
    # --params a,b --values 1,2                 N x M = 1 x 2
    # --params a,b --values 1,2,3               Doesn't work. 
    # --params a,b --values "[1, 2]"            N x M = 1 x 2
    # --params a,b --values "[[1, 2],[3, 4]]"   N x M = 2 x 2
    params = arg_to_list(params, str)
    n_params = len(params)
    values = arg_to_list(values, Union[float, int, str], broadcast=n_params)    # If single value, broadcast to length of number of params.
    # List of lists could be passed directly - do nothing.
    # This handles cases where a single list is passed. 
    # This could either be a single run, multiple params, or multiple runs, single param. How do we know which is which?
    values = arg_to_list(values, List[Union[float, int, str]])           
    if n_params == 1:
        # Transpose values if using a single param - multiple runs.
        values = list(map(list, zip(*values)))
    print(params)
    print(values)

    # Check number of values for each run.
    n_runs = len(values)
    for i in range(n_runs):
        vs = values[i]
        if len(vs) != len(params):
            raise ValueError(f"Got {len(vs)} value/s for run {i} (values={vs}), must match number of params ({len(params)}).")

    # Create runs.
    for i in range(n_runs):
        param_vals = {}
        for j, p in enumerate(params):
            param_vals[p] = values[i][j]
        if dry_run:
            logging.info(f"Would call: {create_slurm_fn.__name__}({args}, {param_vals}, suffix={i}, {kwargs})")
        else:
            create_slurm_fn(*args, **param_vals, suffix=i, **kwargs)

def create_slurm_grid_pmcc(*args, **kwargs) -> None:
    create_slurm_grid(create_slurm_pmcc, *args, **kwargs)

def create_slurm_grid_spartan(*args, **kwargs) -> None:
    create_slurm_grid(create_slurm, *args, **kwargs)

def grid_arg(
    name: str,
    arg_type: Optional[Union[float, int, str]] = None,
    default: Optional[Union[float, int, str]] = None) -> Optional[Union[int, float]]:
    parser = argparse.ArgumentParser()
    if arg_type is None and default is None:
        raise ValueError("Must provide either arg_type or default value - to infer arg type.")
    arg_type = arg_type or type(default)
    parser.add_argument(f'--{name}', type=arg_type, default=default)
    args = parser.parse_args()
    arg = getattr(args, name)
    return arg

PMCC_CPU_PARTITIONS = [
    'debug',
    'rhel_long',
    'rhel_short',
]
PMCC_GPU_PARTITIONS = [
    'rhel_gpu',
]
PMCC_DEFAULT_CPU = 'rhel_short'
PMCC_DEFAULT_GPU = 'rhel_gpu'

def create_slurm_pmcc(
    file: str,
    array: Optional[str] = None,
    enqueue: bool = True,
    memory: int = 128,
    mode: Optional[Literal['cpu', 'gpu']] = None,
    partitions: Optional[str] = None,
    suffix: str = '',
    time: timedelta = 'days:14',
    **kwargs) -> None:
    if 'p' in kwargs or 'partition' in kwargs:
        raise ValueError("Please use the 'partitions' argument to specify partitions, not 'p' or 'partition'.")

    # Determine partitions and mode.
    assert mode in (None, 'cpu', 'gpu'), f"Mode must be one of None, 'cpu', or 'gpu', not {mode}."
    if partitions is None:
        if mode is None:
            # If both are None, default to CPU mode and default CPU partition.
            mode = 'cpu'
            partitions = [PMCC_DEFAULT_CPU]
        else:
            # If mode is specified, set partitions based on mode.
            partitions = [PMCC_DEFAULT_CPU] if mode == 'cpu' else [PMCC_DEFAULT_GPU]
    elif partitions == 'go-fishing':
        mode = 'gpu'
        partitions = PMCC_GPU_PARTITIONS
    else:
        partitions = partitions.split(',')  # Fire won't interpret as a list.

        # Check partitions.
        for p in partitions:
            if p not in PMCC_CPU_PARTITIONS + PMCC_GPU_PARTITIONS:
                raise ValueError(f"Partition {p} not recognised. Must be one of {PMCC_CPU_PARTITIONS + PMCC_GPU_PARTITIONS}.")

        # Set the mode based on requested partitions.
        p0 = partitions[0]
        if p0 in PMCC_CPU_PARTITIONS:
            mode = 'cpu'
        elif p0 in PMCC_GPU_PARTITIONS:
            mode = 'gpu' 

        # Check consistent mode.
        for p in partitions:
            if (mode == 'cpu' and p not in PMCC_CPU_PARTITIONS) or (mode == 'gpu' and p not in PMCC_GPU_PARTITIONS):
                raise ValueError(f"Partitions should be either CPU or GPU, not both.")

    logging.info(f"Queuing job with mode '{mode}' and partitions {partitions}.")

    # Parse time string.
    if isinstance(time, str):
        match = re.match(r'(days:(\d+))?:?(hours:(\d+))?', time)
        days = int(match.group(2)) if match.group(2) is not None else 0
        hours = int(match.group(4)) if match.group(4) is not None else 0
        time = timedelta(days=days, hours=hours)

    for p in partitions:
        # Handle partition specifics.
        p_memory = memory
        p_time = time
        if p == 'rhel_short':
            max_time = timedelta(days=1)
        elif p in ('debug', 'rhel_long', 'rhel_gpu'):
            max_time = timedelta(days=14)
        if p_time > max_time:
            logging.info(f"Partition '{p}' time limit exceeded. Setting time to {max_time}.")
            p_time = max_time

        # Convert time.
        days = p_time.days
        hours, remainder = divmod(p_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Convert kwargs.
        kwarg_str = ''
        for k, v in kwargs.items():
            kwarg_str += f'--{k} "{v}" '

        # Manage resources.
        g_res = """
#SBATCH --gres gpu:1""" if mode == 'gpu' else ''
        c_res = """
#SBATCH --cpus-per-gpu 1""" if mode == 'gpu' else ''

        # Create content.
        content = f"""\
#!/bin/bash
#SBATCH --partition {p}
#SBATCH --mem {p_memory}G{g_res}{c_res}
#SBATCH --time {days}-{hours:02}:{minutes:02}:{seconds:02}

source ~/.bash_profile

python {file} {kwarg_str}
"""

        # Write to file.
        p_file = file.replace('.py', '')
        suffix = f'-{suffix}' if suffix != '' else ''
        filepath = f'{p_file}-{p}{suffix}.slurm'
        print(f"Writing to {filepath}.")
        with open(filepath, 'w') as f:
            f.write(content)

        # Queue job.
        if enqueue:
            command = f'sbatch '
            if array is not None:
                command += f'--array={array} '
            command += f'{filepath}'

            print(f"Queueing job with command: {command}")
            os.system(command)
