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

def create_slurm(
    file: str,
    array: Optional[str] = None,
    memory: int = 128,
    mode: Literal['cpu', 'gpu'] = 'gpu',
    partitions: str = 'gpu-h100',
    queue: bool = True,
    suffix: str = '',
    time: timedelta = 'days:7',
    **kwargs) -> None:
    # Handle arguments.
    if partitions == 'go-fishing':
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
        if queue:
            command = f'sbatch '
            if array is not None:
                command += f'--array={array} '
            command += f'{filepath}'

            print(f"Queueing job with command: {command}")
            os.system(command)

def create_slurm_grid(
    *args,
    params: Union[str, Sequence[str]] = [],     # M params.
    values: Union[int, float, Sequence[Union[int, float]], Sequence[Sequence[Union[int, float]]]] = [],     # N runs.
    **kwargs) -> None:
    # Variations:
    # --params a --values 1                     N x M = 1 x 1
    # --params a --values 1,2                   N x M = 2 x 1
    # --params a,b --values 1,2                 N x M = 1 x 2
    # --params a,b --values 1,2,3               Doesn't work. 
    # --params a,b --values "[1, 2]"            N x M = 1 x 2
    # --params a,b --values "[[1, 2],[3, 4]]"   N x M = 2 x 2
    params = arg_to_list(params, str)
    n_params = len(params)
    values = arg_to_list(values, Union[int, float], broadcast=n_params)    # If single value, broadcast to length of number of params.
    # List of lists could be passed directly - do nothing.
    # This handles cases where a single list is passed. 
    # This could either be a single run, multiple params, or multiple runs, single param. How do we know which is which?
    values = arg_to_list(values, Sequence[Union[int, float]])           
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
        create_slurm(*args, **param_vals, suffix=i, **kwargs)

def grid_arg(
    name: str,
    default: Union[int, float]) -> Union[int, float]:
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--{name}', type=type(default), default=default)
    args = parser.parse_args()
    arg = getattr(args, name)
    return arg

PMCC_CPU_PARTITIONS =[
    'rhel_long',
    'rhel_short',
]
PMCC_GPU_PARTITIONS = [
    'rhel_gpu',
]

def create_slurm_pmcc(
    file: str,
    array: Optional[str] = None,
    memory: int = 128,
    mode: Literal['cpu', 'gpu'] = 'gpu',
    partitions: str = 'rhel_gpu',
    queue: bool = True,
    suffix: str = '',
    time: timedelta = 'days:14',
    **kwargs) -> None:
    # Handle arguments.
    if partitions == 'go-fishing':
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
        if p == 'rhel_short':
            max_time = timedelta(days=1)
        elif p == 'rhel_long' or p == 'rhel_gpu':
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

        # Create content.
        content = f"""\
#!/bin/bash
#SBATCH --partition {p}
#SBATCH --mem {p_memory}G
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
        if queue:
            command = f'sbatch '
            if array is not None:
                command += f'--array={array} '
            command += f'{filepath}'

            print(f"Queueing job with command: {command}")
            os.system(command)
