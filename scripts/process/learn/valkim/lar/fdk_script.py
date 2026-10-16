import subprocess
import sys
import time

log_file = {log_file!r}

# Run FDK. Values are passed as '--opt=value' as e.g. a negative origin would otherwise
# be mistaken for a flag by argparse.
cmd = [
    'rtkfdk',
    '--dimension={recon_size}',
    '--spacing={recon_spacing}',
    '--origin={recon_origin}',
    '-o', {out_file!r},
    '-p', {proj_dir!r},
    '-r', {proj_regexp!r},
    '-g', {geo_file!r},
    '--hardware', 'cuda',
    '--verbose',
    '--pad', '1',
    '--hann', '0.2',
]
cmd_str = subprocess.list2cmdline(cmd)

# Echo output to the console and write it to the log file (replacing any old log).
with open(log_file, 'w') as log:
    print(cmd_str)
    log.write(cmd_str + '\n')
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
    for line in proc.stdout:
        print(line, end='')
        log.write(line)
        log.flush()
    returncode = proc.wait()
    elapsed = time.perf_counter() - start
    elapsed_str = 'Elapsed time: %.2fs (%s).' % (elapsed, time.strftime('%H:%M:%S', time.gmtime(elapsed)))
    print(elapsed_str)
    log.write(elapsed_str + '\n')

sys.exit(returncode)
