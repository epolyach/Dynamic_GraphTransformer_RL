#!/bin/bash

# Start training for medium_gpu_annealing
screen -dmS training_medium_annealing bash -c 'cd /home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL && source venv/bin/activate && python3 training_gpu/scripts/run_training_gpu.py --config configs/medium_gpu_annealing.yaml --model GT+RL --device cuda:0 --force-retrain 2>&1 | tee training_medium_annealing.log'

echo "Training started in screen session 'training_medium_annealing'"
echo ""

# Small delay to avoid potential conflicts
sleep 2

# Start training for medium_gpu_large_model
screen -dmS training_medium_large_model bash -c 'cd /home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL && source venv/bin/activate && python3 training_gpu/scripts/run_training_gpu.py --config configs/medium_gpu_large_model.yaml --model GT+RL --device cuda:0 --force-retrain 2>&1 | tee training_medium_large_model.log'

echo "Training started in screen session 'training_medium_large_model'"
echo ""

# Small delay to avoid potential conflicts
sleep 2

# Start training for medium_gpu_optimal
screen -dmS training_medium_optimal bash -c 'cd /home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL && source venv/bin/activate && python3 training_gpu/scripts/run_training_gpu.py --config configs/medium_gpu_optimal.yaml --model GT+RL --device cuda:0 --force-retrain 2>&1 | tee training_medium_optimal.log'

echo "Training started in screen session 'training_medium_optimal'"
echo ""

echo "========================================="
echo "All training sessions started!"
echo "========================================="
echo ""
echo "Commands for each session:"
echo ""
echo "Medium GPU Annealing:"
echo "  View progress:     screen -r training_medium_annealing"
echo "  Log file:          training_medium_annealing.log"
echo ""
echo "Medium GPU Large Model:"
echo "  View progress:     screen -r training_medium_large_model"
echo "  Log file:          training_medium_large_model.log"
echo ""
echo "Medium GPU Optimal:"
echo "  View progress:     screen -r training_medium_optimal"
echo "  Log file:          training_medium_optimal.log"
echo ""
echo "General commands:"
echo "  Detach:           Ctrl+A, then D"
echo "  List screens:     screen -ls"
echo "  Kill session:     screen -X -S <session_name> quit"
echo ""
echo "Active training screens:"
screen -ls | grep training
