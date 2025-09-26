#!/bin/bash

echo "Starting training_medium_annealing and training_medium_optimal sequentially..."

# Start both trainings sequentially in a screen
screen -dmS training_medium_annealing_optimal bash -c '
cd /home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL
source venv/bin/activate

echo "========================================="
echo "Starting Medium GPU Annealing Training"
echo "========================================="
python3 training_gpu/scripts/run_training_gpu.py --config configs/medium_gpu_annealing.yaml --model GT+RL --device cuda:0 --force-retrain 2>&1 | tee training_medium_annealing_new.log

echo ""
echo "========================================="
echo "Starting Medium GPU Optimal Training"
echo "========================================="
python3 training_gpu/scripts/run_training_gpu.py --config configs/medium_gpu_optimal.yaml --model GT+RL --device cuda:0 --force-retrain 2>&1 | tee training_medium_optimal_new.log

echo ""
echo "========================================="
echo "Both training sessions completed!"
echo "========================================="
'

echo "Training started in screen session 'training_medium_annealing_optimal'"
echo ""
echo "Commands:"
echo "  View progress:     screen -r training_medium_annealing_optimal"
echo "  Detach:           Ctrl+A, then D"
echo "  List screens:     screen -ls"
echo "  Kill if needed:   screen -X -S training_medium_annealing_optimal quit"
echo ""
echo "Log files will be saved as:"
echo "  - training_medium_annealing_new.log"
echo "  - training_medium_optimal_new.log"
