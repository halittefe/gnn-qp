import numpy as np
from pandas import read_csv
import tensorflow as tf
import argparse
import os
from qp_models import QPGNNPolicy

## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="number of training data", default=2000, type=int)
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--embSize", help="embedding size of GNN", default=64, type=int)
parser.add_argument("--epoch", help="max number of epochs", default=1000, type=int)
parser.add_argument("--type", help="model type", default="fea", choices=['fea','obj','sol'])
parser.add_argument("--valSplit", help="fraction of data for validation (0~1)", default=0.2, type=float)
parser.add_argument("--dropout", help="dropout rate", default=0.0, type=float)
parser.add_argument("--weightDecay", help="weight decay for AdamW", default=0.0, type=float)
parser.add_argument("--patience", help="early stopping patience", default=40, type=int)
args = parser.parse_args()

## HELPER FUNCTIONS
def process_train_step(model, dataloader, optimizer, type='fea'):
    c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm, cand_scores = dataloader
    batched_states = (c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm)

    with tf.GradientTape() as tape:
        # training=True => Dropout active
        logits = model(batched_states, training=True)
        loss_tensor = tf.keras.metrics.mean_squared_error(cand_scores, logits)
        loss = tf.reduce_mean(loss_tensor)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    err_rate = None
    if type == "fea":
        # Calculate classification error
        logits_np = logits.numpy()
        cand_scores_np = cand_scores.numpy()
        errs_fp = np.sum((logits_np > 0.5) & (cand_scores_np < 0.5))
        errs_fn = np.sum((logits_np < 0.5) & (cand_scores_np > 0.5))
        total_errs = errs_fp + errs_fn
        err_rate = total_errs / cand_scores_np.shape[0]

    return loss.numpy(), err_rate

def process_eval(model, dataloader, type='fea'):
    c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm, cand_scores = dataloader
    batched_states = (c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm)
    
    # training=False => Dropout inactive
    logits = model(batched_states, training=False)
    loss_tensor = tf.keras.metrics.mean_squared_error(cand_scores, logits)
    loss = tf.reduce_mean(loss_tensor)

    err_rate = None
    if type == "fea":
        # Calculate classification error
        logits_np = logits.numpy()
        cand_scores_np = cand_scores.numpy()
        errs_fp = np.sum((logits_np > 0.5) & (cand_scores_np < 0.5))
        errs_fn = np.sum((logits_np < 0.5) & (cand_scores_np > 0.5))
        total_errs = errs_fp + errs_fn
        err_rate = total_errs / cand_scores_np.shape[0]
    
    return loss.numpy(), err_rate

## SETUP HYPERPARAMETERS
max_epochs = args.epoch
lr = 0.0003
seed = 0
val_split = args.valSplit
weight_decay = args.weightDecay
dropout_rate = args.dropout
patience = args.patience

## DATASET SETUP
trainfolder = "./data-training/"
n_Samples_total = args.data
n_Cons_small = 10   # Each QP has 10 constraints
n_Vars_small = 50   # Each QP has 50 variables
n_Eles_small = 100  # Each QP has 100 nonzeros in matrix A

## LOAD DATASET INTO MEMORY
if args.type == "fea":
    varFeatures_all = read_csv(trainfolder + "VarFeatures_all.csv", header=None).values[:n_Vars_small * n_Samples_total,:]
    conFeatures_all = read_csv(trainfolder + "ConFeatures_all.csv", header=None).values[:n_Cons_small * n_Samples_total,:]
    edgFeatures_all = read_csv(trainfolder + "EdgeFeatures_all.csv", header=None).values[:n_Eles_small * n_Samples_total,:]
    edgIndices_all = read_csv(trainfolder + "EdgeIndices_all.csv", header=None).values[:n_Eles_small * n_Samples_total,:]
    qedgFeatures_all = read_csv(trainfolder + "QEdgeFeatures_all.csv", header=None).values
    qedgIndices_all = read_csv(trainfolder + "QEdgeIndices_all.csv", header=None).values
    labels_all = read_csv(trainfolder + "Labels_feas.csv", header=None).values[:n_Samples_total,:]

elif args.type == "obj":
    varFeatures_all = read_csv(trainfolder + "VarFeatures_feas.csv", header=None).values[:n_Vars_small * n_Samples_total,:]
    conFeatures_all = read_csv(trainfolder + "ConFeatures_feas.csv", header=None).values[:n_Cons_small * n_Samples_total,:]
    edgFeatures_all = read_csv(trainfolder + "EdgeFeatures_feas.csv", header=None).values[:n_Eles_small * n_Samples_total,:]
    edgIndices_all = read_csv(trainfolder + "EdgeIndices_feas.csv", header=None).values[:n_Eles_small * n_Samples_total,:]
    qedgFeatures_all = read_csv(trainfolder + "QEdgeFeatures_feas.csv", header=None).values
    qedgIndices_all = read_csv(trainfolder + "QEdgeIndices_feas.csv", header=None).values
    labels_all = read_csv(trainfolder + "Labels_obj.csv", header=None).values[:n_Samples_total,:]

elif args.type == "sol":
    varFeatures_all = read_csv(trainfolder + "VarFeatures_feas.csv", header=None).values[:n_Vars_small * n_Samples_total,:]
    conFeatures_all = read_csv(trainfolder + "ConFeatures_feas.csv", header=None).values[:n_Cons_small * n_Samples_total,:]
    edgFeatures_all = read_csv(trainfolder + "EdgeFeatures_feas.csv", header=None).values[:n_Eles_small * n_Samples_total,:]
    edgIndices_all = read_csv(trainfolder + "EdgeIndices_feas.csv", header=None).values[:n_Eles_small * n_Samples_total,:]
    qedgFeatures_all = read_csv(trainfolder + "QEdgeFeatures_feas.csv", header=None).values
    qedgIndices_all = read_csv(trainfolder + "QEdgeIndices_feas.csv", header=None).values
    labels_all = read_csv(trainfolder + "Labels_solu.csv", header=None).values[:n_Vars_small * n_Samples_total,:]

nConsF = conFeatures_all.shape[1]
nVarF = varFeatures_all.shape[1]
nEdgeF = edgFeatures_all.shape[1]
nQEdgeF = qedgFeatures_all.shape[1] if qedgFeatures_all.ndim > 1 else 1

## SPLIT DATA INTO TRAIN/VALIDATION SETS
if args.type in ["fea", "obj"]:
    val_count = int(n_Samples_total * val_split)
    train_count = n_Samples_total - val_count

    # Training data
    varFeatures_train = varFeatures_all[:train_count * n_Vars_small, :]
    conFeatures_train = conFeatures_all[:train_count * n_Cons_small, :]
    edgFeatures_train = edgFeatures_all[:train_count * n_Eles_small, :]
    edgIndices_train = edgIndices_all[:train_count * n_Eles_small, :]
    labels_train = labels_all[:train_count, :]

    # Validation data
    varFeatures_val = varFeatures_all[train_count * n_Vars_small:n_Samples_total * n_Vars_small, :]
    conFeatures_val = conFeatures_all[train_count * n_Cons_small:n_Samples_total * n_Cons_small, :]
    edgFeatures_val = edgFeatures_all[train_count * n_Eles_small:n_Samples_total * n_Eles_small, :]
    edgIndices_val = edgIndices_all[train_count * n_Eles_small:n_Samples_total * n_Eles_small, :]
    labels_val = labels_all[train_count:n_Samples_total, :]

    # For simplicity, use all quadratic edges for all instances
    # In a real implementation, you would split these appropriately too
    qedgFeatures_train = qedgFeatures_all
    qedgIndices_train = qedgIndices_all
    qedgFeatures_val = qedgFeatures_all
    qedgIndices_val = qedgIndices_all

elif args.type == "sol":
    val_count = int(n_Samples_total * val_split)
    train_count = n_Samples_total - val_count

    val_count_vars = val_count * n_Vars_small
    train_count_vars = train_count * n_Vars_small

    # Training data
    varFeatures_train = varFeatures_all[:train_count_vars, :]
    conFeatures_train = conFeatures_all[:train_count * n_Cons_small, :]
    edgFeatures_train = edgFeatures_all[:train_count * n_Eles_small, :]
    edgIndices_train = edgIndices_all[:train_count * n_Eles_small, :]
    labels_train = labels_all[:train_count_vars, :]

    # Validation data
    varFeatures_val = varFeatures_all[train_count_vars:, :]
    conFeatures_val = conFeatures_all[train_count * n_Cons_small:, :]
    edgFeatures_val = edgFeatures_all[train_count * n_Eles_small:, :]
    edgIndices_val = edgIndices_all[train_count * n_Eles_small:, :]
    labels_val = labels_all[train_count_vars:, :]

    # For simplicity, use all quadratic edges for all instances
    qedgFeatures_train = qedgFeatures_all
    qedgIndices_train = qedgIndices_all
    qedgFeatures_val = qedgFeatures_all
    qedgIndices_val = qedgIndices_all

## SETUP MODEL AND SAVED MODEL PATH
if not os.path.exists('./saved-models/'):
    os.makedirs('./saved-models/')
model_path = './saved-models/qp_' + args.type + '_d' + str(n_Samples_total) + '_s' + str(args.embSize) + '.pkl'

## SETUP TENSORFLOW GPU
tf.random.set_seed(seed)
gpu_index = int(args.gpu)
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

## MAIN TRAINING LOOP
with tf.device("GPU:" + str(gpu_index) if len(gpus) > 0 else "/CPU:0"):
    # Convert data to TensorFlow tensors
    varFeatures_train_tf = tf.constant(varFeatures_train, dtype=tf.float32)
    conFeatures_train_tf = tf.constant(conFeatures_train, dtype=tf.float32)
    edgFeatures_train_tf = tf.constant(edgFeatures_train, dtype=tf.float32)
    edgIndices_train_tf = tf.transpose(tf.constant(edgIndices_train, dtype=tf.int32))
    qedgFeatures_train_tf = tf.constant(qedgFeatures_train, dtype=tf.float32)
    qedgIndices_train_tf = tf.transpose(tf.constant(qedgIndices_train, dtype=tf.int32))
    labels_train_tf = tf.constant(labels_train, dtype=tf.float32)

    train_data = (
        conFeatures_train_tf, edgIndices_train_tf, edgFeatures_train_tf,
        varFeatures_train_tf, qedgIndices_train_tf, qedgFeatures_train_tf,
        conFeatures_train.shape[0], varFeatures_train.shape[0],
        n_Cons_small, n_Vars_small, labels_train_tf
    )

    varFeatures_val_tf = tf.constant(varFeatures_val, dtype=tf.float32)
    conFeatures_val_tf = tf.constant(conFeatures_val, dtype=tf.float32)
    edgFeatures_val_tf = tf.constant(edgFeatures_val, dtype=tf.float32)
    edgIndices_val_tf = tf.transpose(tf.constant(edgIndices_val, dtype=tf.int32))
    qedgFeatures_val_tf = tf.constant(qedgFeatures_val, dtype=tf.float32)
    qedgIndices_val_tf = tf.transpose(tf.constant(qedgIndices_val, dtype=tf.int32))
    labels_val_tf = tf.constant(labels_val, dtype=tf.float32)

    val_data = (
        conFeatures_val_tf, edgIndices_val_tf, edgFeatures_val_tf,
        varFeatures_val_tf, qedgIndices_val_tf, qedgFeatures_val_tf,
        conFeatures_val.shape[0], varFeatures_val.shape[0],
        n_Cons_small, n_Vars_small, labels_val_tf
    )

    # Create model
    if args.type == "sol":
        model = QPGNNPolicy(
            embSize=args.embSize,
            nConsF=nConsF,
            nEdgeF=nEdgeF,
            nVarF=nVarF,
            nQEdgeF=nQEdgeF,
            isGraphLevel=False,
            dropout_rate=dropout_rate
        )
    else:
        model = QPGNNPolicy(
            embSize=args.embSize,
            nConsF=nConsF,
            nEdgeF=nEdgeF,
            nVarF=nVarF,
            nQEdgeF=nQEdgeF,
            isGraphLevel=True,
            dropout_rate=dropout_rate
        )

    # Setup optimizer - use Adam since AdamW might not be available in older TF versions
    try:
        # Try using AdamW if available (TF 2.11+)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    except AttributeError:
        # Fall back to standard Adam if AdamW is not available
        print("AdamW optimizer not available, using standard Adam optimizer")
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # Note: weight decay will not be applied

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    wait = 0
    best_epoch = 0

    # Training loop
    for epoch in range(max_epochs):
        # Train step
        train_loss, train_err = process_train_step(model, train_data, optimizer, type=args.type)
        
        # Validation step
        val_loss, val_err = process_eval(model, val_data, type=args.type)
        
        # Print progress
        if args.type == "fea":
            print(f"Epoch {epoch:4d}: Train Loss={train_loss:.6f}, Train Err={train_err:.4f}, "
                  f"Val Loss={val_loss:.6f}, Val Err={val_err:.4f}")
        else:
            print(f"Epoch {epoch:4d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        # Check for improvement and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_epoch = epoch
            model.save_state(model_path)
            print(f"  ✓ Saved best model at epoch {epoch} with val_loss={val_loss:.6f}")
        else:
            wait += 1
            if wait >= patience:
                print(f"  ✗ Early stopping after {patience} epochs without improvement")
                print(f"  ✓ Best model was at epoch {best_epoch} with val_loss={best_val_loss:.6f}")
                break

    print(f"Training completed. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")




