# This script generates random QP instances for training and testing using OSQP solver
import numpy as np
import random as rd
import os
import argparse
from pandas import read_csv
from scipy import linalg
from sklearn.datasets import make_spd_matrix
import osqp
from scipy import sparse

## ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--k_train", default='200')
parser.add_argument("--k_test", default='100')
parser.add_argument("--m", default='10')
parser.add_argument("--n", default='50')
parser.add_argument("--nnz", default='100')
parser.add_argument("--qsparsity", default='0.1')
parser.add_argument("--prob", default="0.3")
parser.add_argument("--reg", default="1e-6", help="Regularization for Q matrix")
parser.add_argument("--check_rank", action="store_true", help="Check constraint matrix rank")
args = parser.parse_args()

## SETUP
k_data_training = int(args.k_train)    # number of training data 
k_data_testing = int(args.k_test)      # number of testing data
m = int(args.m)                        # number of constraints
n = int(args.n)                        # number of variables
nnz = int(args.nnz)                    # number of nonzero elements in A
q_sparsity = float(args.qsparsity)     # sparsity of Q matrix (ratio of non-zeros)
prob_equal = float(args.prob)          # probability for equality constraints
reg_param = float(args.reg)            # regularization parameter for Q matrix
folder_training = "./data-training"    # folder to save training data 
folder_testing = "./data-testing"      # folder to save testing data
check_rank = args.check_rank           # whether to check the rank of constraint matrix

def make_sparse_psd(n, density=0.1, random_state=None, reg=1e-6):
    """Create a sparse positive semi-definite matrix with improved conditioning"""
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate a sparse lower triangular matrix
    mask = np.random.rand(n, n) < density
    mask = np.tril(mask)
    L = np.random.randn(n, n) * mask
    
    # Create PSD matrix using L*L.T
    Q = L @ L.T
    
    # Add stronger regularization to diagonal to ensure positive definiteness
    min_eig = np.min(np.real(np.linalg.eigvals(Q)))
    if min_eig < 0:
        Q += (-min_eig + reg) * np.eye(n)
    else:
        Q += reg * np.eye(n)
    
    # Make it sparser if needed
    if density < 0.5:
        mask = np.random.rand(n, n) < density * 2
        mask = (mask & mask.T) | np.eye(n, dtype=bool)  # Keep diagonal
        Q = Q * mask
    
    # Scale the matrix to have reasonable magnitude
    Q = Q / (np.max(np.abs(Q)) + 1e-8)
    
    return Q

def check_matrix_rank(A, tolerance=1e-10):
    """Check the rank of a matrix and return if it's full rank"""
    # Compute SVD
    u, s, vh = linalg.svd(A)
    
    # Count number of non-zero singular values
    rank = np.sum(s > tolerance)
    full_rank = rank == min(A.shape)
    
    # Calculate condition number to assess numerical stability
    if len(s) > 0:
        cond = s[0] / s[-1] if s[-1] > tolerance else float('inf')
    else:
        cond = float('inf')
        
    return full_rank, rank, cond

def solve_qp_osqp(Q, c, A_ub, b_ub, A_eq, b_eq, bounds):
    """
    QP problemi çözmek için OSQP kullanır
    """
    n = Q.shape[0]
    
    try:
        # OSQP için gerekli matrisleri hazırla
        # Seyrek matris format dönüşümlerini kontrol et
        if not sparse.issparse(Q):
            P = sparse.csc_matrix(Q)
        else:
            P = Q.tocsc()
        
        # Problem ölçeklendirmesi
        scale = max(1.0, np.max(np.abs(P.data)))
        P = P / scale
        q = c / scale
        
        # Kısıtlamaları birleştir
        constraint_matrices = []
        lower_bounds = []
        upper_bounds = []
        
        # Eşitsizlik kısıtları
        if len(A_ub) > 0:
            A_ub_sparse = sparse.csc_matrix(A_ub)
            constraint_matrices.append(A_ub_sparse)
            lower_bounds.append(-np.inf * np.ones(len(b_ub)))
            upper_bounds.append(b_ub)
        
        # Eşitlik kısıtları
        if len(A_eq) > 0:
            A_eq_sparse = sparse.csc_matrix(A_eq)
            constraint_matrices.append(A_eq_sparse)
            lower_bounds.append(b_eq)
            upper_bounds.append(b_eq)
        
        # Sınır kısıtları
        if len(bounds) > 0:
            # Değişken sınırları için bir kimlik matrisi
            A_bounds = sparse.eye(n, format='csc')
            constraint_matrices.append(A_bounds)
            lower_bounds.append(bounds[:, 0])
            upper_bounds.append(bounds[:, 1])
        
        # Kısıtlar varsa, hepsini birleştir
        if constraint_matrices:
            A = sparse.vstack(constraint_matrices, format='csc')
            l = np.hstack(lower_bounds)
            u = np.hstack(upper_bounds)
        else:
            A = sparse.csc_matrix((0, n))
            l = np.array([])
            u = np.array([])
        
        # OSQP ayarları - daha sağlam çözüm için
        settings = {
            'verbose': False,
            'eps_abs': 1e-3,
            'eps_rel': 1e-3,
            'eps_prim_inf': 1e-4,  # Primal infeasibility tolerance
            'eps_dual_inf': 1e-4,  # Dual infeasibility tolerance
            'max_iter': 8000,
            'polish': True,
            'polish_refine_iter': 3,
            'warm_start': True,
            'adaptive_rho': True,
            'adaptive_rho_interval': 25
        }
        
        # OSQP problemi oluştur
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, **settings)
        
        # Problemi çöz
        results = prob.solve()
        
        # Çözüm başarılı mı kontrol et
        if results.info.status in ['solved', 'solved inaccurate']:
            # Ölçeklendirmeyi geri al
            x = results.x
            obj_val = results.info.obj_val * scale
            return x, True, obj_val
        else:
            print(f"OSQP status: {results.info.status}")
            return None, False, None
    
    except Exception as e:
        print(f"OSQP exception: {e}")
        return None, False, None

def generateQP(k_data, configs, folder):
    '''
    This function generates and saves QP instances using OSQP solver.
    - k_data: the number of instances to generate 
    - configs: (m,n,nnz,q_sparsity,prob_equal), configurations of each QP instance 
    - folder: the folder to save those generated QPs
    '''
    m, n, nnz, q_sparsity, prob_equal = configs
    successful_count = 0
    problem_index = 0
    
    while successful_count < k_data:
        path = folder + "/Data" + str(successful_count)
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Randomly sample a QP problem
        # min 0.5*x^T Q x + c^T x
        # s.t. Aub x <= bub, Aeq x = beq, lb <= x <= ub
        
        # Generate Q (positive semidefinite matrix)
        Q = make_sparse_psd(n, density=q_sparsity, random_state=problem_index, reg=reg_param)
        
        # Generate other components
        c = np.random.uniform(-1, 1, n) * 0.01
        b = np.random.uniform(-1, 1, m)
        
        bounds = np.random.normal(0, 10, size=(n, 2))
        
        # Ensure lower bounds < upper bounds
        for j in range(n):
            if bounds[j, 0] > bounds[j, 1]:
                bounds[j, 0], bounds[j, 1] = bounds[j, 1], bounds[j, 0]
            
        # Generate constraint matrix A
        A = np.zeros((m, n))
        EdgeIndex = np.zeros((nnz, 2))
        EdgeIndex1D = rd.sample(range(m * n), nnz)
        EdgeFeature = np.random.normal(0, 1, nnz)
        
        for l in range(nnz):
            i = int(EdgeIndex1D[l] / n)
            j = EdgeIndex1D[l] - i * n
            EdgeIndex[l, 0] = i
            EdgeIndex[l, 1] = j
            A[i, j] = EdgeFeature[l]

        # Classify constraints
        circ = np.random.binomial(1, prob_equal, size=m)  # 1 means = constraint, 0 means <= constraint
        A_ub = A[circ == 0, :]
        b_ub = b[circ == 0]
        A_eq = A[circ == 1, :]
        b_eq = b[circ == 1]
        
        # Check constraint matrix rank if requested
        proceed = True
        if check_rank and len(A_eq) > 0:
            full_rank, rank, cond = check_matrix_rank(A_eq)
            if not full_rank or cond > 1e10:  # If not full rank or ill-conditioned
                proceed = False
                problem_index += 1
                continue
        
        # Generate Q edges (variable-to-variable connections)
        Q_edges = []
        Q_weights = []
        for i in range(n):
            for j in range(n):
                if abs(Q[i,j]) > 1e-8:  # Non-zero entry
                    Q_edges.append((i, j))
                    Q_weights.append(Q[i,j])
        
        Q_edges = np.array(Q_edges)
        Q_weights = np.array(Q_weights)
        
        # Solve the QP problem using OSQP
        try:
            x, is_feasible, obj_val = solve_qp_osqp(Q, c, A_ub, b_ub, A_eq, b_eq, bounds)
            
            # Çözüm None ise (hata oluşmuşsa) problemi atla
            if x is None:
                problem_index += 1
                continue
                
            is_bounded = True  # PSD Q ile garanti edilir
            
        except Exception as e:
            print(f"Error solving QP instance {problem_index}: {e}")
            problem_index += 1
            continue
        
        # Write to CSV files
        np.savetxt(path + '/ConFeatures.csv', np.hstack((b.reshape(m, 1), circ.reshape(m, 1))), delimiter=',', fmt='%10.5f')
        np.savetxt(path + '/EdgeFeatures.csv', EdgeFeature, fmt='%10.5f')
        np.savetxt(path + '/EdgeIndices.csv', EdgeIndex, delimiter=',', fmt='%d')
        np.savetxt(path + '/VarFeatures.csv', np.hstack((c.reshape(n, 1), bounds)), delimiter=',', fmt='%10.5f')
        np.savetxt(path + '/QEdgeIndices.csv', Q_edges, delimiter=',', fmt='%d')
        np.savetxt(path + '/QEdgeFeatures.csv', Q_weights, fmt='%10.5f')
        np.savetxt(path + '/Labels_feas.csv', [int(is_feasible)], fmt='%d')
        
        if is_feasible and is_bounded:
            np.savetxt(path + '/Labels_obj.csv', [obj_val], fmt='%10.5f')
            np.savetxt(path + '/Labels_solu.csv', x, fmt='%10.5f')
            successful_count += 1
        else:
            # If not feasible, we don't count it and try again
            os.system(f"rm -rf {path}")  # Remove the directory
        
        problem_index += 1
        
        if successful_count % 100 == 0 and successful_count > 0:
            print(f'Generated: {successful_count}, Attempted: {problem_index}')

def combineGraphsAll(k_data, configs, folder):
    '''
    This function combines all QP instances in "folder" to a large graph to facilitate training.
    This function also makes labels for the feasibility of QP instances.
    '''
    m, n, nnz, q_sparsity, prob_equal = configs

    # First, count the total number of Q edges
    total_q_edges = 0
    for k in range(k_data):
        LPfolder = folder + "/Data" + str(k)
        if os.path.exists(LPfolder + "/QEdgeIndices.csv"):
            q_edges = read_csv(LPfolder + "/QEdgeIndices.csv", header=None).values
            total_q_edges += len(q_edges)

    # Pre-allocate arrays
    ConFeatures_all = np.zeros((k_data * m, 2))
    EdgeFeatures_all = np.zeros((k_data * nnz, 1))
    EdgeIndices_all = np.zeros((k_data * nnz, 2))
    QEdgeFeatures_all = np.zeros((total_q_edges, 1))
    QEdgeIndices_all = np.zeros((total_q_edges, 2))
    VarFeatures_all = np.zeros((k_data * n, 3))
    Labels_feas = np.zeros((k_data, 1))

    # Now combine all graphs
    q_edge_offset = 0
    for k in range(k_data):
        LPfolder = folder + "/Data" + str(k)
        
        if not os.path.exists(LPfolder + "/ConFeatures.csv"):
            continue
            
        varFeatures = read_csv(LPfolder + "/VarFeatures.csv", header=None).values
        conFeatures = read_csv(LPfolder + "/ConFeatures.csv", header=None).values
        edgeFeatures = read_csv(LPfolder + "/EdgeFeatures.csv", header=None).values
        edgeIndices = read_csv(LPfolder + "/EdgeIndices.csv", header=None).values
        
        # Q edges
        if os.path.exists(LPfolder + "/QEdgeIndices.csv"):
            qEdgeIndices = read_csv(LPfolder + "/QEdgeIndices.csv", header=None).values
            qEdgeFeatures = read_csv(LPfolder + "/QEdgeFeatures.csv", header=None).values
            
            # Adjust indices for the combined graph
            qEdgeIndices[:, 0] = qEdgeIndices[:, 0] + k * n
            qEdgeIndices[:, 1] = qEdgeIndices[:, 1] + k * n
            
            num_q_edges = len(qEdgeIndices)
            QEdgeFeatures_all[q_edge_offset:q_edge_offset+num_q_edges, :] = qEdgeFeatures.reshape(-1, 1)
            QEdgeIndices_all[q_edge_offset:q_edge_offset+num_q_edges, :] = qEdgeIndices
            q_edge_offset += num_q_edges
        
        # Adjust constraint-variable edge indices
        edgeIndices[:, 0] = edgeIndices[:, 0] + k * m
        edgeIndices[:, 1] = edgeIndices[:, 1] + k * n
        
        # Feasibility label
        labelsFeas = read_csv(LPfolder + "/Labels_feas.csv", header=None).values
        
        # Store in the combined arrays
        ConFeatures_all[range(k * m, (k + 1) * m), :] = conFeatures
        VarFeatures_all[range(k * n, (k + 1) * n), :] = varFeatures
        EdgeFeatures_all[range(k * nnz, (k + 1) * nnz), :] = edgeFeatures.reshape(-1, 1)
        EdgeIndices_all[range(k * nnz, (k + 1) * nnz), :] = edgeIndices
        Labels_feas[k] = labelsFeas

        if k % 100 == 0:
            print("Combined:", k)
            
    # Save combined features and labels
    np.savetxt(folder + '/ConFeatures_all.csv', ConFeatures_all, delimiter=',', fmt='%10.5f')
    np.savetxt(folder + '/EdgeFeatures_all.csv', EdgeFeatures_all, fmt='%10.5f')
    np.savetxt(folder + '/EdgeIndices_all.csv', EdgeIndices_all, delimiter=',', fmt='%d')
    np.savetxt(folder + '/QEdgeFeatures_all.csv', QEdgeFeatures_all[:q_edge_offset], fmt='%10.5f')
    np.savetxt(folder + '/QEdgeIndices_all.csv', QEdgeIndices_all[:q_edge_offset], delimiter=',', fmt='%d')
    np.savetxt(folder + '/VarFeatures_all.csv', VarFeatures_all, delimiter=',', fmt='%10.5f')
    np.savetxt(folder + '/Labels_feas.csv', Labels_feas, fmt='%10.5f')

def combineGraphsFeas(k_data, configs, folder):
    '''
    This function combines all feasible QP instances in "folder".
    This function also makes labels for the optimal objective and optimal solution.
    '''
    m, n, nnz, q_sparsity, prob_equal = configs

    # Collect info: which QP instances are feasible
    k_list = []
    total_q_edges = 0
    
    for k in range(k_data):
        LPfolder = folder + "/Data" + str(k)
        if os.path.exists(LPfolder + '/Labels_solu.csv'):
            k_list.append(k)
            if os.path.exists(LPfolder + "/QEdgeIndices.csv"):
                q_edges = read_csv(LPfolder + "/QEdgeIndices.csv", header=None).values
                total_q_edges += len(q_edges)

    k_feas = len(k_list)
    if k_feas == 0:
        print("No feasible QP instances found!")
        return

    # Pre-allocate arrays
    ConFeatures_feas = np.zeros((k_feas * m, 2))
    EdgeFeatures_feas = np.zeros((k_feas * nnz, 1))
    EdgeIndices_feas = np.zeros((k_feas * nnz, 2))
    QEdgeFeatures_feas = np.zeros((total_q_edges, 1))
    QEdgeIndices_feas = np.zeros((total_q_edges, 2))
    VarFeatures_feas = np.zeros((k_feas * n, 3))
    Labels_solu = np.zeros((k_feas * n, 1))
    Labels_obj = np.zeros((k_feas, 1))

    # Now combine all feasible graphs
    q_edge_offset = 0
    for l, k in enumerate(k_list):
        LPfolder = folder + "/Data" + str(k)
        
        varFeatures = read_csv(LPfolder + "/VarFeatures.csv", header=None).values
        conFeatures = read_csv(LPfolder + "/ConFeatures.csv", header=None).values
        edgeFeatures = read_csv(LPfolder + "/EdgeFeatures.csv", header=None).values
        edgeIndices = read_csv(LPfolder + "/EdgeIndices.csv", header=None).values
        
        # Q edges
        if os.path.exists(LPfolder + "/QEdgeIndices.csv"):
            qEdgeIndices = read_csv(LPfolder + "/QEdgeIndices.csv", header=None).values
            qEdgeFeatures = read_csv(LPfolder + "/QEdgeFeatures.csv", header=None).values
            
            # Adjust indices for the combined graph
            qEdgeIndices[:, 0] = qEdgeIndices[:, 0] + l * n
            qEdgeIndices[:, 1] = qEdgeIndices[:, 1] + l * n
            
            num_q_edges = len(qEdgeIndices)
            QEdgeFeatures_feas[q_edge_offset:q_edge_offset+num_q_edges, :] = qEdgeFeatures.reshape(-1, 1)
            QEdgeIndices_feas[q_edge_offset:q_edge_offset+num_q_edges, :] = qEdgeIndices
            q_edge_offset += num_q_edges
        
        # Adjust constraint-variable edge indices
        edgeIndices[:, 0] = edgeIndices[:, 0] + l * m
        edgeIndices[:, 1] = edgeIndices[:, 1] + l * n
        
        # Labels
        labelsSolu = read_csv(LPfolder + "/Labels_solu.csv", header=None).values
        labelsObj = read_csv(LPfolder + "/Labels_obj.csv", header=None).values
        
        # Store in the combined arrays
        ConFeatures_feas[range(l * m, (l + 1) * m), :] = conFeatures
        VarFeatures_feas[range(l * n, (l + 1) * n), :] = varFeatures
        EdgeFeatures_feas[range(l * nnz, (l + 1) * nnz), :] = edgeFeatures.reshape(-1, 1)
        EdgeIndices_feas[range(l * nnz, (l + 1) * nnz), :] = edgeIndices
        Labels_solu[range(l * n, (l + 1) * n), :] = labelsSolu.reshape(-1, 1)
        Labels_obj[l] = labelsObj

        if l % 100 == 0:
            print("Combined:", l, '/', k_feas)
    
    # Save combined features and labels
    np.savetxt(folder + '/ConFeatures_feas.csv', ConFeatures_feas, delimiter=',', fmt='%10.5f')
    np.savetxt(folder + '/EdgeFeatures_feas.csv', EdgeFeatures_feas, fmt='%10.5f')
    np.savetxt(folder + '/EdgeIndices_feas.csv', EdgeIndices_feas, delimiter=',', fmt='%d')
    np.savetxt(folder + '/QEdgeFeatures_feas.csv', QEdgeFeatures_feas[:q_edge_offset], fmt='%10.5f')
    np.savetxt(folder + '/QEdgeIndices_feas.csv', QEdgeIndices_feas[:q_edge_offset], delimiter=',', fmt='%d')
    np.savetxt(folder + '/VarFeatures_feas.csv', VarFeatures_feas, delimiter=',', fmt='%10.5f')
    np.savetxt(folder + '/Labels_solu.csv', Labels_solu, fmt='%10.5f')
    np.savetxt(folder + '/Labels_obj.csv', Labels_obj, fmt='%10.5f')

## MAIN SCRIPT
if not os.path.exists(folder_training):
    os.makedirs(folder_training)
if not os.path.exists(folder_testing):
    os.makedirs(folder_testing)

print("Generating training data.")
generateQP(k_data_training, (m, n, nnz, q_sparsity, prob_equal), folder_training)
combineGraphsAll(k_data_training, (m, n, nnz, q_sparsity, prob_equal), folder_training)
combineGraphsFeas(k_data_training, (m, n, nnz, q_sparsity, prob_equal), folder_training)

print("Generating testing data.")
generateQP(k_data_testing, (m, n, nnz, q_sparsity, prob_equal), folder_testing)
combineGraphsAll(k_data_testing, (m, n, nnz, q_sparsity, prob_equal), folder_testing)
combineGraphsFeas(k_data_testing, (m, n, nnz, q_sparsity, prob_equal), folder_testing) 