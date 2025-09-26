#!/bin/bash

echo "========================================="
echo "Launching Sequential Medium GPU Training"
echo "========================================="

# Kill the existing large model session first
echo "Stopping existing training_medium_large_model session..."
screen -X -S training_medium_large_model quit 2>/dev/null
sleep 3

# Start the sequential training script in a screen session
screen -dmS medium_training_sequential bash -c './run_all_medium_trainings_sequential.sh'

echo ""
echo "Sequential training started in screen session 'medium_training_sequential'"
echo ""
echo "Commands:"
echo "  View progress:     screen -r medium_training_sequential"
echo "  Detach:           Ctrl+A, then D"
echo "  List screens:     screen -ls"
echo "  Kill if needed:   screen -X -S medium_training_sequential quit"
echo ""
echo "Training order:"
echo "  1. medium_gpu_annealing       -> training_medium_annealing_seq.log"
echo "  2. medium_gpu_large_model     -> training_medium_large_model_seq.log"
echo "  3. medium_gpu_optimal         -> training_medium_optimal_seq.log"
echo ""
echo "All trainings will run sequentially to avoid GPU memory conflicts."
echo ""
echo "Active screens:"
screen -ls | grep medium
