#!/bin/bash

# Function to run the script
run_script() {
    python optuna_daw_dorsomedial.py  # Replace with the actual script name and its parameters if needed
}

# Number of times to run the script
num_runs=10

# Loop to run the script n times
for ((i=1; i<=$num_runs; i++)); do
    run_script &  # Run the script in the background
    wait         # Wait for the background process to finish before starting the next one
done

echo "All scripts have completed."