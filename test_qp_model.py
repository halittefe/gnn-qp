# Part 1 of QP GNN testing script
import numpy as np
from pandas import read_csv
import tensorflow as tf
import argparse
import os
from qp_models import QPGNNPolicy

## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="number of training data", default=1000)
parser.add_argument("--dataTest", help="number of test data", default=4000)
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--embSize", help="embedding size of GNN", default="64")
parser.add_argument("--type", help="what's the type of the model", default="sol", choices=['fea','obj','sol'])
parser.add_argument("--set", help="which set you want to test on?", default="test", choices=['test','train'])
parser.add_argument("--loss", help="loss function used in testing", default="l2", choices=['mse','l2'])
args = parser.parse_args()

## FUNCTION OF TESTING
def process(model, dataloader, type='fea', loss='mse', n_Vars_small=50):
    c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm, cand_scores = dataloader
    batched_states = (c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm)  
    logits = model(batched_states, tf.convert_to_tensor(False)) 
    
    return_err = None
    
    if type == "fea":
        logits_np = logits.numpy().flatten()
        cand_scores_np = cand_scores.numpy().flatten()
        pred_labels = (logits_np > 0.5).astype(int)
        true_labels = (cand_scores_np > 0.5).astype(int)
        tp = np.sum((pred_labels == 1) & (true_labels == 1))
        tn = np.sum((pred_labels == 0) & (true_labels == 0))
        fp = np.sum((pred_labels == 1) & (true_labels == 0))
        fn = np.sum((pred_labels == 0) & (true_labels == 1))
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(true_labels, logits_np)
        except ImportError:
            auc = None
        err_rate = (fp + fn) / cand_scores_np.shape[0]
        return_err = err_rate
        return return_err, auc, tp, tn, fp, fn
    
    if type == "obj":
        if loss == 'mse':
            loss = tf.keras.metrics.mean_squared_error(cand_scores, logits)
            return_err = tf.reduce_mean(loss).numpy()
        else:
            loss = (tf.abs(cand_scores - logits) / (tf.abs(cand_scores) + 1.0))
            return_err = tf.reduce_mean(loss).numpy()
            
    if type == "sol":
        if loss == 'mse':
            loss = tf.keras.metrics.mean_squared_error(cand_scores, logits)
            return_err = tf.reduce_mean(loss).numpy()
        else:
            length_sol = logits.shape[0]
            cand_scores = tf.reshape(cand_scores, [int(length_sol/n_Vars_small), n_Vars_small])
            logits = tf.reshape(logits, [int(length_sol/n_Vars_small), n_Vars_small])
            loss = tf.math.reduce_euclidean_norm(cand_scores - logits, axis=1)
            norm = tf.math.reduce_euclidean_norm(cand_scores, axis=1) + 1.0
            return_err = tf.reduce_mean(loss / norm).numpy()

    return return_err

## SET-UP DATASET
datafolder = "./data-training/" if args.set == "train" else "./data-testing/"
n_Samples_test = int(args.dataTest)
n_Cons_small = 10  # Each QP has 10 constraints
n_Vars_small = 50  # Each QP has 50 variables
n_Eles_small = 100 # Each QP has 100 nonzeros in matrix A

## SET-UP MODEL
embSize = int(args.embSize)
n_Samples = int(args.data)
model_path = '/home/halit/GNN-QP/saved-models/qp_' + args.type + '_d' + str(n_Samples) + '_s' + str(embSize) + '.pkl'

# Part 2 of QP GNN testing script
# This script should be run after test_qp_model_part1.py

## LOAD DATASET INTO MEMORY
if args.type == "fea":
    varFeatures = read_csv(datafolder + "VarFeatures_all.csv", header=None).values[:n_Vars_small * n_Samples_test,:]
    conFeatures = read_csv(datafolder + "ConFeatures_all.csv", header=None).values[:n_Cons_small * n_Samples_test,:]
    edgFeatures = read_csv(datafolder + "EdgeFeatures_all.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
    edgIndices = read_csv(datafolder + "EdgeIndices_all.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
    qedgFeatures = read_csv(datafolder + "QEdgeFeatures_all.csv", header=None).values
    qedgIndices = read_csv(datafolder + "QEdgeIndices_all.csv", header=None).values
    labels = read_csv(datafolder + "Labels_feas.csv", header=None).values[:n_Samples_test,:]
elif args.type == "obj":
    varFeatures = read_csv(datafolder + "VarFeatures_feas.csv", header=None).values[:n_Vars_small * n_Samples_test,:]
    conFeatures = read_csv(datafolder + "ConFeatures_feas.csv", header=None).values[:n_Cons_small * n_Samples_test,:]
    edgFeatures = read_csv(datafolder + "EdgeFeatures_feas.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
    edgIndices = read_csv(datafolder + "EdgeIndices_feas.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
    qedgFeatures = read_csv(datafolder + "QEdgeFeatures_feas.csv", header=None).values
    qedgIndices = read_csv(datafolder + "QEdgeIndices_feas.csv", header=None).values
    labels = read_csv(datafolder + "Labels_obj.csv", header=None).values[:n_Samples_test,:]
elif args.type == "sol":
    varFeatures = read_csv(datafolder + "VarFeatures_feas.csv", header=None).values[:n_Vars_small * n_Samples_test,:]
    conFeatures = read_csv(datafolder + "ConFeatures_feas.csv", header=None).values[:n_Cons_small * n_Samples_test,:]
    edgFeatures = read_csv(datafolder + "EdgeFeatures_feas.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
    edgIndices = read_csv(datafolder + "EdgeIndices_feas.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
    qedgFeatures = read_csv(datafolder + "QEdgeFeatures_feas.csv", header=None).values
    qedgIndices = read_csv(datafolder + "QEdgeIndices_feas.csv", header=None).values
    labels = read_csv(datafolder + "Labels_solu.csv", header=None).values[:n_Vars_small * n_Samples_test,:]

nConsF = conFeatures.shape[1]
nVarF = varFeatures.shape[1]
nEdgeF = edgFeatures.shape[1]
nQEdgeF = qedgFeatures.shape[1] if qedgFeatures.ndim > 1 else 1
n_Cons = conFeatures.shape[0]
n_Vars = varFeatures.shape[0]

## SET-UP TENSORFLOW
gpu_index = int(args.gpu)
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

with tf.device("GPU:"+str(gpu_index)):
    ### LOAD DATASET INTO GPU ###
    varFeatures = tf.constant(varFeatures, dtype=tf.float32)
    conFeatures = tf.constant(conFeatures, dtype=tf.float32)
    edgFeatures = tf.constant(edgFeatures, dtype=tf.float32)
    edgIndices = tf.constant(edgIndices, dtype=tf.int32)
    edgIndices = tf.transpose(edgIndices)
    qedgFeatures = tf.constant(qedgFeatures, dtype=tf.float32)
    qedgIndices = tf.constant(qedgIndices, dtype=tf.int32)
    qedgIndices = tf.transpose(qedgIndices)
    labels = tf.constant(labels, dtype=tf.float32)
    
    data = (conFeatures, edgIndices, edgFeatures, varFeatures, qedgIndices, qedgFeatures, 
           n_Cons, n_Vars, n_Cons_small, n_Vars_small, labels)

    ### LOAD MODEL ###
    if args.type == "sol":
        model = QPGNNPolicy(embSize, nConsF, nEdgeF, nVarF, nQEdgeF, isGraphLevel=False, dropout_rate=0.0)
    else:
        model = QPGNNPolicy(embSize, nConsF, nEdgeF, nVarF, nQEdgeF, dropout_rate=0.0)
    
    model.restore_state(model_path)

    # Build the model by calling it once with sample input (fixes summary() error)
    _ = model((conFeatures, edgIndices, edgFeatures, varFeatures, qedgIndices, qedgFeatures, n_Cons, n_Vars, n_Cons_small, n_Vars_small), training=False)

    ### TEST MODEL ###
    err = process(model, data, type=args.type, loss=args.loss, n_Vars_small=n_Vars_small)
    try:
        model.summary()
    except ValueError as e:
        print(f"Warning: model.summary() failed: {e}")
    print(f"MODEL: {model_path}, DATA-SET: {datafolder}, NUM-DATA: {n_Samples_test}, LOSS: {args.loss}, ERR: {err}")