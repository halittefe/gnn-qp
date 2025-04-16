import numpy as np
import scipy.optimize as opt
import random as rd
import os
import argparse
from pandas import read_csv
from scipy import linalg

############################
#   ARGUMENT PARSING
############################
parser = argparse.ArgumentParser()
parser.add_argument("--ftrain", default='1000', help="Number of feasible training examples")
parser.add_argument("--itrain", default='1000', help="Number of infeasible training examples")
parser.add_argument("--ftest", default='400', help="Number of feasible test examples")
parser.add_argument("--itest", default='400', help="Number of infeasible test examples")
parser.add_argument("--m", default='10')
parser.add_argument("--n", default='50')
parser.add_argument("--nnz", default='100')
parser.add_argument("--qsparsity", default='0.1')
parser.add_argument("--prob", default="0.3")
parser.add_argument("--reg", default="1e-6", help="Regularization for Q matrix")
parser.add_argument("--check_rank", action="store_true", help="Check constraint matrix rank (optional)")
args = parser.parse_args()

############################
#   BASIC SETTINGS
############################
feasible_train = int(args.ftrain)
infeasible_train = int(args.itrain)
feasible_test = int(args.ftest)
infeasible_test = int(args.itest)
m = int(args.m)                        # number of constraints
n = int(args.n)                        # number of variables
nnz = int(args.nnz)                    # number of nonzero elements in A
q_sparsity = float(args.qsparsity)     # sparsity of Q matrix (ratio of non-zeros)
prob_equal = float(args.prob)          # probability for equality constraints
reg_param = float(args.reg)            # regularization parameter for Q matrix
folder_training = "./data-training"    # folder to save training data
folder_testing = "./data-testing"      # folder to save testing data
check_rank = args.check_rank           # whether to check the rank of constraint matrix

############################
#   HELPER FUNCTIONS
############################
def make_sparse_psd(n, density=0.1, random_state=None, reg=1e-6):
    """
    Create a sparse positive semi-definite matrix with some regularization on diagonal.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate a sparse lower triangular matrix
    mask = np.random.rand(n, n) < density
    mask = np.tril(mask)
    L = np.random.randn(n, n) * mask
    
    # Create PSD matrix Q = L * L^T
    Q = L @ L.T
    
    # Add regularization to diagonal
    Q += reg * np.eye(n)
    
    # Optional extra sparsification step if needed
    if density < 0.5:
        mask2 = np.random.rand(n, n) < (density * 2)
        mask2 = (mask2 & mask2.T) | np.eye(n, dtype=bool)
        Q = Q * mask2
    
    return Q

def check_matrix_rank(A, tolerance=1e-10):
    """
    Check the rank of a matrix, return (full_rank, rank, condition_number).
    """
    u, s, vh = linalg.svd(A)
    rank = np.sum(s > tolerance)
    full_rank = (rank == min(A.shape))
    
    cond = float('inf')
    if len(s) > 0 and s[-1] > tolerance:
        cond = s[0] / s[-1]
        
    return full_rank, rank, cond

############################
#   COMBINE FUNCTIONS
############################
def combineGraphsAll(k_data, configs, folder):
    """
    Combine all QP instances (feasible + infeasible) into single set of:
      - VarFeatures_all.csv
      - ConFeatures_all.csv
      - EdgeFeatures_all.csv
      - EdgeIndices_all.csv
      - QEdgeFeatures_all.csv
      - QEdgeIndices_all.csv
      - Labels_feas.csv  (0/1)
    """
    m, n, nnz, q_sparsity, prob_equal = configs

    # First, count total Q edges
    total_q_edges = 0
    for k in range(k_data):
        LPfolder = os.path.join(folder, f"Data{k}")
        if os.path.exists(os.path.join(LPfolder, "QEdgeIndices.csv")):
            q_edges = read_csv(os.path.join(LPfolder, "QEdgeIndices.csv"), header=None).values
            total_q_edges += len(q_edges)

    # Pre-allocate
    ConFeatures_all = np.zeros((k_data * m, 2))
    EdgeFeatures_all = np.zeros((k_data * nnz, 1))
    EdgeIndices_all = np.zeros((k_data * nnz, 2))
    QEdgeFeatures_all = np.zeros((total_q_edges, 1))
    QEdgeIndices_all = np.zeros((total_q_edges, 2))
    VarFeatures_all = np.zeros((k_data * n, 3))
    Labels_feas = np.zeros((k_data, 1))

    q_edge_offset = 0
    valid_count = 0
    for k in range(k_data):
        LPfolder = os.path.join(folder, f"Data{k}")
        feas_path = os.path.join(LPfolder, "Labels_feas.csv")
        if not os.path.exists(feas_path):
            # Some folder might have been removed if it didn't meet the feasibility target, skip it
            continue
        
        conFeatures = read_csv(os.path.join(LPfolder, "ConFeatures.csv"), header=None).values
        varFeatures = read_csv(os.path.join(LPfolder, "VarFeatures.csv"), header=None).values
        edgeFeatures = read_csv(os.path.join(LPfolder, "EdgeFeatures.csv"), header=None).values
        edgeIndices = read_csv(os.path.join(LPfolder, "EdgeIndices.csv"), header=None).values
        labelsFeas = read_csv(feas_path, header=None).values
        
        # Q edges
        qEdgeIndices = None
        qEdgeFeatures = None
        if os.path.exists(os.path.join(LPfolder, "QEdgeIndices.csv")):
            qEdgeIndices = read_csv(os.path.join(LPfolder, "QEdgeIndices.csv"), header=None).values
            qEdgeFeatures = read_csv(os.path.join(LPfolder, "QEdgeFeatures.csv"), header=None).values

        # Adjust constraint-variable edge indices
        edgeIndices[:, 0] = edgeIndices[:, 0] + valid_count * m
        edgeIndices[:, 1] = edgeIndices[:, 1] + valid_count * n

        ConFeatures_all[valid_count*m : (valid_count+1)*m, :] = conFeatures
        VarFeatures_all[valid_count*n : (valid_count+1)*n, :] = varFeatures
        EdgeFeatures_all[valid_count*nnz : (valid_count+1)*nnz, :] = edgeFeatures.reshape(-1, 1)
        EdgeIndices_all[valid_count*nnz : (valid_count+1)*nnz, :] = edgeIndices
        Labels_feas[valid_count] = labelsFeas
        
        # QEdge
        if qEdgeIndices is not None and len(qEdgeIndices) > 0:
            qEdgeIndices[:, 0] = qEdgeIndices[:, 0] + valid_count * n
            qEdgeIndices[:, 1] = qEdgeIndices[:, 1] + valid_count * n
            num_q_edges = len(qEdgeIndices)
            QEdgeFeatures_all[q_edge_offset:q_edge_offset+num_q_edges, :] = qEdgeFeatures.reshape(-1, 1)
            QEdgeIndices_all[q_edge_offset:q_edge_offset+num_q_edges, :] = qEdgeIndices
            q_edge_offset += num_q_edges

        valid_count += 1
        if valid_count % 100 == 0:
            print("Combined (all):", valid_count)
    
    # Kırpma
    QEdgeFeatures_all = QEdgeFeatures_all[:q_edge_offset]
    QEdgeIndices_all = QEdgeIndices_all[:q_edge_offset]

    # Save combined
    np.savetxt(os.path.join(folder, 'VarFeatures_all.csv'), VarFeatures_all[:valid_count*n, :], delimiter=',', fmt='%10.5f')
    np.savetxt(os.path.join(folder, 'ConFeatures_all.csv'), ConFeatures_all[:valid_count*m, :], delimiter=',', fmt='%10.5f')
    np.savetxt(os.path.join(folder, 'EdgeFeatures_all.csv'), EdgeFeatures_all[:valid_count*nnz, :], fmt='%10.5f')
    np.savetxt(os.path.join(folder, 'EdgeIndices_all.csv'), EdgeIndices_all[:valid_count*nnz, :], delimiter=',', fmt='%d')
    np.savetxt(os.path.join(folder, 'QEdgeFeatures_all.csv'), QEdgeFeatures_all, fmt='%10.5f')
    np.savetxt(os.path.join(folder, 'QEdgeIndices_all.csv'), QEdgeIndices_all, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(folder, 'Labels_feas.csv'), Labels_feas[:valid_count], fmt='%d')

    print("combineGraphsAll finished. Kept:", valid_count, "instances total.")

def combineGraphsFeas(k_data, configs, folder):
    """
    Combine all *feasible* QP instances in 'folder':
      - VarFeatures_feas.csv
      - ConFeatures_feas.csv
      - EdgeFeatures_feas.csv
      - EdgeIndices_feas.csv
      - QEdgeFeatures_feas.csv
      - QEdgeIndices_feas.csv
      - Labels_solu.csv (optimal solutions)
      - Labels_obj.csv  (optimal objectives)
    """
    m, n, nnz, q_sparsity, prob_equal = configs
    feasible_indices = []
    total_q_edges = 0

    # İlk geçiş: hangi Data klasöründe feasible var?
    for k in range(k_data):
        LPfolder = os.path.join(folder, f"Data{k}")
        feas_path = os.path.join(LPfolder, "Labels_feas.csv")
        solu_path = os.path.join(LPfolder, "Labels_solu.csv")
        if os.path.exists(feas_path) and os.path.exists(solu_path):
            # Yani feasible bir problem
            feasible_indices.append(k)
            # Q-edge sayısı
            if os.path.exists(os.path.join(LPfolder, "QEdgeIndices.csv")):
                q_edges = read_csv(os.path.join(LPfolder, "QEdgeIndices.csv"), header=None).values
                total_q_edges += len(q_edges)

    k_feas = len(feasible_indices)

    # Pre-allocate
    ConFeatures_feas = np.zeros((k_feas * m, 2))
    EdgeFeatures_feas = np.zeros((k_feas * nnz, 1))
    EdgeIndices_feas = np.zeros((k_feas * nnz, 2))
    QEdgeFeatures_feas = np.zeros((total_q_edges, 1))
    QEdgeIndices_feas = np.zeros((total_q_edges, 2))
    VarFeatures_feas = np.zeros((k_feas * n, 3))
    Labels_solu = np.zeros((k_feas * n, 1))
    Labels_obj = np.zeros((k_feas, 1))

    q_edge_offset = 0
    for idx, k in enumerate(feasible_indices):
        LPfolder = os.path.join(folder, f"Data{k}")
        conFeatures = read_csv(os.path.join(LPfolder, "ConFeatures.csv"), header=None).values
        varFeatures = read_csv(os.path.join(LPfolder, "VarFeatures.csv"), header=None).values
        edgeFeatures = read_csv(os.path.join(LPfolder, "EdgeFeatures.csv"), header=None).values
        edgeIndices = read_csv(os.path.join(LPfolder, "EdgeIndices.csv"), header=None).values
        labelsSolu = read_csv(os.path.join(LPfolder, "Labels_solu.csv"), header=None).values
        labelsObj = read_csv(os.path.join(LPfolder, "Labels_obj.csv"), header=None).values
        
        # Q edges
        qEdgeIndices = None
        qEdgeFeatures = None
        if os.path.exists(os.path.join(LPfolder, "QEdgeIndices.csv")):
            qEdgeIndices = read_csv(os.path.join(LPfolder, "QEdgeIndices.csv"), header=None).values
            qEdgeFeatures = read_csv(os.path.join(LPfolder, "QEdgeFeatures.csv"), header=None).values
            
        # Adjust edge indices
        edgeIndices[:, 0] = edgeIndices[:, 0] + idx * m
        edgeIndices[:, 1] = edgeIndices[:, 1] + idx * n
        
        ConFeatures_feas[idx*m : (idx+1)*m, :] = conFeatures
        VarFeatures_feas[idx*n : (idx+1)*n, :] = varFeatures
        EdgeFeatures_feas[idx*nnz : (idx+1)*nnz, :] = edgeFeatures.reshape(-1, 1)
        EdgeIndices_feas[idx*nnz : (idx+1)*nnz, :] = edgeIndices
        Labels_solu[idx*n : (idx+1)*n, 0] = labelsSolu.reshape(-1)
        Labels_obj[idx] = labelsObj
        
        # Q-edge indices
        if qEdgeIndices is not None and len(qEdgeIndices) > 0:
            qEdgeIndices[:, 0] = qEdgeIndices[:, 0] + idx * n
            qEdgeIndices[:, 1] = qEdgeIndices[:, 1] + idx * n
            num_q_edges = len(qEdgeIndices)
            QEdgeFeatures_feas[q_edge_offset : q_edge_offset+num_q_edges, 0] = qEdgeFeatures.reshape(-1)
            QEdgeIndices_feas[q_edge_offset : q_edge_offset+num_q_edges, :] = qEdgeIndices
            q_edge_offset += num_q_edges

        if idx % 100 == 0:
            print("Combined (feas):", idx, "/", k_feas)
    
    # Kırpma
    QEdgeFeatures_feas = QEdgeFeatures_feas[:q_edge_offset]
    QEdgeIndices_feas = QEdgeIndices_feas[:q_edge_offset]

    # Kaydet
    np.savetxt(os.path.join(folder, 'VarFeatures_feas.csv'), VarFeatures_feas, delimiter=',', fmt='%10.5f')
    np.savetxt(os.path.join(folder, 'ConFeatures_feas.csv'), ConFeatures_feas, delimiter=',', fmt='%10.5f')
    np.savetxt(os.path.join(folder, 'EdgeFeatures_feas.csv'), EdgeFeatures_feas, fmt='%10.5f')
    np.savetxt(os.path.join(folder, 'EdgeIndices_feas.csv'), EdgeIndices_feas, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(folder, 'QEdgeFeatures_feas.csv'), QEdgeFeatures_feas, fmt='%10.5f')
    np.savetxt(os.path.join(folder, 'QEdgeIndices_feas.csv'), QEdgeIndices_feas, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(folder, 'Labels_solu.csv'), Labels_solu, fmt='%10.5f')
    np.savetxt(os.path.join(folder, 'Labels_obj.csv'), Labels_obj, fmt='%10.5f')

    print("combineGraphsFeas finished. Feasible count:", k_feas)

############################
#   BALANCED DATA GENERATION
############################
def generateQP_balanced(feasible_target, infeasible_target, configs, folder):
    """
    İstenen sayıda feasible ve infeasible QP örneği üretir.
    :param feasible_target: Kaç tane feasible problem üretilecek
    :param infeasible_target: Kaç tane infeasible problem üretilecek
    :param configs: (m,n,nnz,q_sparsity,prob_equal)
    :param folder: Kayıtların yazılacağı klasör
    """
    import numpy as np
    import random as rd
    import os
    from pandas import read_csv
    
    m, n, nnz, q_sparsity, prob_equal = configs
    feasible_count = 0
    infeasible_count = 0
    total_count = 0  # Üretilen problem sayısı (deneme sayısı)

    # Hedeflere ulaşana kadar döngü
    while feasible_count < feasible_target or infeasible_count < infeasible_target:
        
        path = os.path.join(folder, f"Data{total_count}")
        if not os.path.exists(path):
            os.makedirs(path)

        # Rastgele QP oluştur
        # 1) Q matrisi
        Q = make_sparse_psd(n, density=q_sparsity, random_state=total_count, reg=reg_param)

        # 2) c
        c = np.random.uniform(-1, 1, n) * 0.01

        # 3) b
        b = np.random.uniform(-1, 1, m)

        # 4) Bounds
        bounds = np.random.normal(0, 10, size=(n, 2))
        for j in range(n):
            if bounds[j, 0] > bounds[j, 1]:
                bounds[j, 0], bounds[j, 1] = bounds[j, 1], bounds[j, 0]

        # 5) A
        A = np.zeros((m, n))
        EdgeIndex = np.zeros((nnz, 2))
        EdgeIndex1D = rd.sample(range(m * n), nnz)
        EdgeFeature = np.random.normal(0, 1, nnz)
        for l in range(nnz):
            i_ = int(EdgeIndex1D[l] / n)
            j_ = EdgeIndex1D[l] - i_ * n
            EdgeIndex[l, 0] = i_
            EdgeIndex[l, 1] = j_
            A[i_, j_] = EdgeFeature[l]

        # Eşitlik/ Eşitsizlik ayrımı
        circ = np.random.binomial(1, prob_equal, size=m)  # 1 => eşitlik, 0 => <=
        A_ub = A[circ == 0, :]
        b_ub = b[circ == 0]
        A_eq = A[circ == 1, :]
        b_eq = b[circ == 1]
        
        # Eğer check_rank isteniyorsa:
        if check_rank and len(A_eq) > 0:
            full_rank, rank_val, cond_val = check_matrix_rank(A_eq)
            if (not full_rank) or (cond_val > 1e10):
                # Belki bu örneği çöpe atabiliriz?
                os.system(f"rm -rf {path}")
                total_count += 1
                continue
        
        # QEdges bul
        Q_edges = []
        Q_weights = []
        for i in range(n):
            for j in range(n):
                if abs(Q[i,j]) > 1e-8:
                    Q_edges.append((i, j))
                    Q_weights.append(Q[i,j])
        Q_edges = np.array(Q_edges)
        Q_weights = np.array(Q_weights)

        # QP'yi çözmeyi dene
        is_feasible = False
        result = None
        try:
            x0 = np.mean(bounds, axis=1)  # baslangic noktasi
            constraints = []
            # Eşitsizlik
            if len(A_ub) > 0:
                for row_i in range(len(b_ub)):
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, rr=row_i: b_ub[rr] - np.dot(A_ub[rr, :], x)
                    })
            # Eşitlik
            if len(A_eq) > 0:
                for row_i in range(len(b_eq)):
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda x, rr=row_i: np.dot(A_eq[rr, :], x) - b_eq[rr]
                    })
            
            result = opt.minimize(
                lambda x: 0.5*np.dot(x, np.dot(Q, x)) + np.dot(c,x),
                x0,
                method='SLSQP',
                jac=lambda x: np.dot(Q, x) + c,
                bounds=[(bounds[i,0], bounds[i,1]) for i in range(n)],
                constraints=constraints,
                options={'ftol':1e-8, 'disp': False, 'maxiter':500}
            )
            
            if result.success:
                is_feasible = True
        except:
            pass

        # CSV'lere kaydet (problem verileri her halükarda)
        np.savetxt(os.path.join(path, 'ConFeatures.csv'),
                   np.hstack((b.reshape(m, 1), circ.reshape(m, 1))),
                   delimiter=',', fmt='%10.5f')
        
        np.savetxt(os.path.join(path, 'EdgeFeatures.csv'),
                   EdgeFeature, fmt='%10.5f')
        
        np.savetxt(os.path.join(path, 'EdgeIndices.csv'),
                   EdgeIndex, delimiter=',', fmt='%d')
        
        np.savetxt(os.path.join(path, 'VarFeatures.csv'),
                   np.hstack((c.reshape(n,1), bounds)),
                   delimiter=',', fmt='%10.5f')
        
        np.savetxt(os.path.join(path, 'QEdgeIndices.csv'),
                   Q_edges, delimiter=',', fmt='%d')
        
        np.savetxt(os.path.join(path, 'QEdgeFeatures.csv'),
                   Q_weights, fmt='%10.5f')

        if is_feasible:
            # Feasible
            if feasible_count < feasible_target:
                # Kaydet
                np.savetxt(os.path.join(path, 'Labels_feas.csv'), [1], fmt='%d')
                np.savetxt(os.path.join(path, 'Labels_obj.csv'), [result.fun], fmt='%10.5f')
                np.savetxt(os.path.join(path, 'Labels_solu.csv'), result.x, fmt='%10.5f')
                feasible_count += 1
            else:
                # Feasible limitine ulaştık, bu örneği silelim veya infeasible'a dönüştürmek isterseniz vs.
                os.system(f"rm -rf {path}")
                total_count += 1
                continue
        else:
            # Infeasible veya çözemedi
            if infeasible_count < infeasible_target:
                np.savetxt(os.path.join(path, 'Labels_feas.csv'), [0], fmt='%d')
                # obj, solu yok
                infeasible_count += 1
            else:
                # Infeasible limitine ulaştık
                os.system(f"rm -rf {path}")
                total_count += 1
                continue

        total_count += 1
        
        # Progress
        if (feasible_count + infeasible_count) % 50 == 0:
            print(f"[BALANCED GEN] Feasible so far: {feasible_count}, Infeasible so far: {infeasible_count}, total used: {feasible_count + infeasible_count}")

    print(f"Balanced dataset generation done. Produced {feasible_count} feasible and {infeasible_count} infeasible in {folder}.")


############################
#   MAIN SCRIPT
############################
if __name__ == "__main__":
    if not os.path.exists(folder_training):
        os.makedirs(folder_training)
    if not os.path.exists(folder_testing):
        os.makedirs(folder_testing)

    # CONFIG tuple
    configs = (m, n, nnz, q_sparsity, prob_equal)

    # 1) Generate TRAIN data (balanced)
    print("Generating training data (balanced).")
    generateQP_balanced(feasible_train, infeasible_train, configs, folder_training)
    combineGraphsAll(feasible_train + infeasible_train, configs, folder_training)
    combineGraphsFeas(feasible_train + infeasible_train, configs, folder_training)

    # 2) Generate TEST data (balanced)
    print("Generating testing data (balanced).")
    generateQP_balanced(feasible_test, infeasible_test, configs, folder_testing)
    combineGraphsAll(feasible_test + infeasible_test, configs, folder_testing)
    combineGraphsFeas(feasible_test + infeasible_test, configs, folder_testing)

    print("All done.")
