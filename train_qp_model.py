# This script trains GNN models for QP problems
import numpy as np
from pandas import read_csv
import tensorflow as tf
import argparse
import os
from qp_models import QPGNNPolicy
## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="number of training data", default=1000)
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--embSize", help="embedding size of GNN", default="64")
parser.add_argument("--epoch", help="num of epoch", default="10000")
parser.add_argument("--type", help="what's the type of the model", default="fea", choices=['fea','obj','sol'])
args = parser.parse_args()

## FUNCTION OF TRAINING PER EPOCH
def process(model, dataloader, optimizer, type='fea'):
    c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm, cand_scores = dataloader
    batched_states = (c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm)  

    with tf.GradientTape() as tape:
        logits = model(batched_states, tf.convert_to_tensor(True)) 
        loss = tf.keras.metrics.mean_squared_error(cand_scores, logits)
        loss = tf.reduce_mean(loss)
    
    grads = tape.gradient(target=loss, sources=model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))

    return_loss = loss.numpy()
    errs = None
    err_rate = None
    
    if type == "fea":
        errs_fp = np.sum((logits.numpy() > 0.5) & (cand_scores.numpy() < 0.5))
        errs_fn = np.sum((logits.numpy() < 0.5) & (cand_scores.numpy() > 0.5))
        errs = errs_fp + errs_fn
        err_rate = errs / cand_scores.shape[0]

    return return_loss, errs, err_rate

## SET-UP HYPER PARAMETERS
max_epochs = int(args.epoch)
lr = 0.0003
seed = 0

## SET-UP DATASET
trainfolder = "./data-training/"
n_Samples = int(args.data)
n_Cons_small = 10  # Each QP has 10 constraints
n_Vars_small = 50  # Each QP has 50 variables
n_Eles_small = 100 # Each QP has 100 nonzeros in matrix A

## SET-UP MODEL
embSize = int(args.embSize)
if not os.path.exists('./saved-models/'):
    os.mkdir('./saved-models/')
model_path = './saved-models/qp_' + args.type + '_d' + str(n_Samples) + '_s' + str(embSize) + '.pkl'

## LOAD DATASET INTO MEMORY
if args.type == "fea":
    varFeatures = read_csv(trainfolder + "VarFeatures_all.csv", header=None).values[:n_Vars_small * n_Samples,:]
    conFeatures = read_csv(trainfolder + "ConFeatures_all.csv", header=None).values[:n_Cons_small * n_Samples,:]
    edgFeatures = read_csv(trainfolder + "EdgeFeatures_all.csv", header=None).values[:n_Eles_small * n_Samples,:]
    edgIndices = read_csv(trainfolder + "EdgeIndices_all.csv", header=None).values[:n_Eles_small * n_Samples,:]
    qedgFeatures = read_csv(trainfolder + "QEdgeFeatures_all.csv", header=None).values
    qedgIndices = read_csv(trainfolder + "QEdgeIndices_all.csv", header=None).values
    labels = read_csv(trainfolder + "Labels_feas.csv", header=None).values[:n_Samples,:]
elif args.type == "obj":
    varFeatures = read_csv(trainfolder + "VarFeatures_feas.csv", header=None).values[:n_Vars_small * n_Samples,:]
    conFeatures = read_csv(trainfolder + "ConFeatures_feas.csv", header=None).values[:n_Cons_small * n_Samples,:]
    edgFeatures = read_csv(trainfolder + "EdgeFeatures_feas.csv", header=None).values[:n_Eles_small * n_Samples,:]
    edgIndices = read_csv(trainfolder + "EdgeIndices_feas.csv", header=None).values[:n_Eles_small * n_Samples,:]
    qedgFeatures = read_csv(trainfolder + "QEdgeFeatures_feas.csv", header=None).values
    qedgIndices = read_csv(trainfolder + "QEdgeIndices_feas.csv", header=None).values
    labels = read_csv(trainfolder + "Labels_obj.csv", header=None).values[:n_Samples,:]
elif args.type == "sol":
    varFeatures = read_csv(trainfolder + "VarFeatures_feas.csv", header=None).values[:n_Vars_small * n_Samples,:]
    conFeatures = read_csv(trainfolder + "ConFeatures_feas.csv", header=None).values[:n_Cons_small * n_Samples,:]
    edgFeatures = read_csv(trainfolder + "EdgeFeatures_feas.csv", header=None).values[:n_Eles_small * n_Samples,:]
    edgIndices = read_csv(trainfolder + "EdgeIndices_feas.csv", header=None).values[:n_Eles_small * n_Samples,:]
    qedgFeatures = read_csv(trainfolder + "QEdgeFeatures_feas.csv", header=None).values
    qedgIndices = read_csv(trainfolder + "QEdgeIndices_feas.csv", header=None).values
    labels = read_csv(trainfolder + "Labels_solu.csv", header=None).values[:n_Vars_small * n_Samples,:]

nConsF = conFeatures.shape[1]
nVarF = varFeatures.shape[1]
nEdgeF = edgFeatures.shape[1]
nQEdgeF = qedgFeatures.shape[1] if qedgFeatures.ndim > 1 else 1
n_Cons = conFeatures.shape[0]
n_Vars = varFeatures.shape[0]

## SET-UP TENSORFLOW
tf.random.set_seed(seed)
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
    
    train_data = (conFeatures, edgIndices, edgFeatures, varFeatures, qedgIndices, qedgFeatures, 
                 n_Cons, n_Vars, n_Cons_small, n_Vars_small, labels)

    ### INITIALIZATION ###
    if args.type == "sol":
        model = QPGNNPolicy(embSize, nConsF, nEdgeF, nVarF, nQEdgeF, isGraphLevel=False)
    else:
        model = QPGNNPolicy(embSize, nConsF, nEdgeF, nVarF, nQEdgeF)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_init, _, _ = process(model, train_data, optimizer, type=args.type)
    epoch = 0
    count_restart = 0
    err_best = 2
    loss_best = 1e10
    
    ### MAIN LOOP ###
    while epoch <= max_epochs:
        train_loss, errs, err_rate = process(model, train_data, optimizer, type=args.type)
            
        if args.type == "fea":
            print(f"EPOCH: {epoch}, TRAIN LOSS: {train_loss}, ERRS: {errs}, ERRATE: {err_rate}")
            if err_rate < err_best:
                model.save_state(model_path)
                print("model saved to:", model_path)
                err_best = err_rate
        else:
            print(f"EPOCH: {epoch}, TRAIN LOSS: {train_loss}")
            if train_loss < loss_best:
                model.save_state(model_path)
                print("model saved to:", model_path)
                loss_best = train_loss
        
        ## If the loss does not go down, we restart the training to re-try another initialization.
        if epoch == 200 and count_restart < 3 and (train_loss > loss_init * 0.8 or (err_rate is not None and err_rate > 0.5)):
            print("Fail to reduce loss, restart...")
            if args.type == "sol":
                model = QPGNNPolicy(embSize, nConsF, nEdgeF, nVarF, nQEdgeF, isGraphLevel=False)
            else:
                model = QPGNNPolicy(embSize, nConsF, nEdgeF, nVarF, nQEdgeF)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            loss_init, _, _ = process(model, train_data, optimizer, type=args.type)
            epoch = 0
            count_restart += 1
            
        epoch += 1
    
    print("Count of restart:", count_restart)
    model.summary()