#!/bin/bash

# This script runs the QP experiments to reproduce the results in the paper

# Generate data
echo "Generating data for QP experiments..."
python generate_qp_data.py --k_train 200 --k_test 100

# Train models for different tasks and embedding sizes
echo "Training feasibility prediction models..."
python train_qp_model.py --type fea --data 10 --embSize 64
python train_qp_model.py --type fea --data 50 --embSize 64
python train_qp_model.py --type fea --data 250 --embSize 64

echo "Training objective value prediction models..."
python train_qp_model.py --type obj --data 100 --embSize 64
python train_qp_model.py --type obj --data 500 --embSize 64
python train_qp_model.py --type obj --data 2500 --embSize 64

echo "Training solution prediction models..."
python train_qp_model.py --type sol --data 100 --embSize 128
python train_qp_model.py --type sol --data 500 --embSize 128
python train_qp_model.py --type sol --data 2500 --embSize 128

# Test models on training and test sets
echo "Testing models on training set..."
python test_all_qp_models.py --type fea --set train --loss l2
python test_all_qp_models.py --type obj --set train --loss l2
python test_all_qp_models.py --type sol --set train --loss l2

echo "Testing models on test set..."
python test_all_qp_models.py --type fea --set test --loss l2 --dataTest 1000
python test_all_qp_models.py --type obj --set test --loss l2 --dataTest 1000
python test_all_qp_models.py --type sol --set test --loss l2 --dataTest 1000

echo "All experiments completed!"