#!/bin/bash

# Get total_items from embedded Python
total_items=$(python3 - <<EOF
import json
with open('files/structures_dict_modified.json') as f:
    data = json.load(f)
print(len(data))
EOF
)

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_DIR="$SCRIPT_DIR/files/logs" # subdirectory path
mkdir -p "$LOG_DIR" # Create it if it doesn't exist

batch_size=$((total_items / 50))
echo "Total items: $total_items"
echo "Batch size: $batch_size"

batch_size=25
total_items=50

max_jobs=4
job_count=0

for ((i=0; i<total_items; i+=batch_size)); do
    start=$i
    end=$((i + batch_size))
    output_file="$LOG_DIR/run_from${start}to${end}.log"
    echo "$output_file"
    python test_bondstruct.py "$start" "$end" > "$output_file" 2>&1 &
    ((job_count++))
    if ((job_count % max_jobs == 0)); then
        wait  # Wait for current batch of jobs to finish
    fi
done

wait  # Final wait to catch any remaining jobs

echo "All batches completed."
