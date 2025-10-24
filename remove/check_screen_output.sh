#!/bin/bash
# Capture last 20 lines from screen session without attaching
screen -S training_optimal -X hardcopy /tmp/training_optimal_output.txt
if [ -f /tmp/training_optimal_output.txt ]; then
    echo "=== Last output from training_optimal ==="
    tail -20 /tmp/training_optimal_output.txt | grep -E "Epoch|time=|Total|EET" || tail -10 /tmp/training_optimal_output.txt
fi

screen -S training_optimal_T20 -X hardcopy /tmp/training_optimal_T20_output.txt
if [ -f /tmp/training_optimal_T20_output.txt ]; then
    echo -e "\n=== Last output from training_optimal_T20 ==="
    tail -20 /tmp/training_optimal_T20_output.txt | grep -E "Epoch|time=|Total|EET" || tail -10 /tmp/training_optimal_T20_output.txt
fi
