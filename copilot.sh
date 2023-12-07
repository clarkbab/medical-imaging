
# Set the highest priority for a specific SLURM job or jobs
scontrol update jobid=<job_id> Priority=1

# Remove job dependencies for all SLURM jobs
for job_id in $(squeue -h -o "%i"); do
    scontrol update jobid=$job_id Dependency=
done

