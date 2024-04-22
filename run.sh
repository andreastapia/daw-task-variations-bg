#!/bin/bash

# Number of times to execute the script sequentially
total_executions=20

# Path to your Python script
script_to_execute="main.py"

# Loop to execute the script sequentially
for ((i=1; i<=$total_executions; i++))
do
    echo "executing simulation N°$i"
    python "basal_ganglia/$script_to_execute" -e doll-ventral-fixed #change experiment depending on simulation declared in basal_ganglia/main.py
done

echo "All script executions have been completed."