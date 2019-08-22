# (@TODO: PEP8 documentation)
# This is just a more modular version of the notebook
# Please refer to the notebook for more thorough description
# of the computation / the equations
import numpy as np
import pandas as pd
import torch
import random
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

def dataset_drop_labels(y, features_nb, labels_nb, drop_rate = 0.4):
    dropped_y_modified_data_df = y.copy()
    random_selections = [None] * y.shape[0]
    for i in tqdm(range(y.shape[0])):
        dropped_y_modified_data_df.iloc[i, features_nb + np.random.choice(labels_nb, int(labels_nb * drop_rate))] = 0
    return dropped_y_modified_data_df

def computing_V_X(xs_train_np):
    V_X = []
    rank_one_array = []
    xs_train_20_nearest = []
    xs_train_7th = []
    x_indices = np.array([range(xs_train_np.shape[0])])
    xs_train_with_idx = np.concatenate((xs_train_np,  x_indices.T), axis=1)

    # Computing 20'th neighbours
    for current in xs_train_with_idx:
        order_values = [(None, None)] * 20
        for other in xs_train_with_idx:
            if current[-1] != other[-1]:
                euclidean = (current[:-1] - other[:-1]) ** 2
                d_square = sum(euclidean)
                idx = other[-1]
                shifting = False
                temporary = (idx, d_square)
                for i in range(len(order_values)):
                    if shifting:
                        current_temp = order_values[i]
                        order_values[i] = temporary
                        temporary = current_temp
                        continue
                    if order_values[i][0] == None:
                        order_values[i] = (idx, d_square)
                        break
                    if order_values[i][1] >= d_square:
                        temporary = order_values[i]
                        order_values[i] = (idx, d_square)
                        shifting = True
        xs_train_7th.append(order_values[6])
        xij = [0.0] * xs_train_with_idx.shape[0]
        for e in order_values:
            xij[int(e[0])] = e[1]
        V_X.append(xij)
    xs_train_20_nearest.append([round(e[1], 4) for e in order_values])
    # Adjusting, normalizing by 7th neighbour
    V_X = np.array(V_X)
    for i in tqdm(range(V_X.shape[0])):
        for j in range(V_X.shape[1]):
            sigma_i = xs_train_7th[i][1]
            sigma_j = xs_train_7th[j][1]
            V_X[i][j] /= (sigma_i * sigma_j)
    # Turning to a density?
    V_X = np.exp(-V_X)
    V_X = torch.from_numpy(V_X).float()
    return V_X

def computing_D_X(V_X):
    D_X = []
    for v_x_i in V_X:
        D_X.append(sum(v_x_i))
    D_X = torch.diag(torch.Tensor(D_X).float())
    return D_X

def computing_L_X(V_X, D_X):
    D_X_sqrt_inverse = torch.sqrt(torch.inverse(D_X))
    L_X = torch.eye(D_X.shape[0]) - torch.mm(D_X_sqrt_inverse, torch.mm(V_X, D_X_sqrt_inverse))
    return L_X

def Vc_distance(Y_subvector_i, Y_subvector_j, n=10):
    norm_i = torch.norm(Y_subvector_i)
    norm_j = torch.norm(Y_subvector_j)
    denominator = norm_i * norm_j
    m_square_matrix = torch.dot(Y_subvector_i, Y_subvector_j)
    return torch.exp(-n * (1 - m_square_matrix / denominator))

def show_normalized_heatmap(V_C):
    import matplotlib.pyplot as plt
    import seaborn as sns; sns.set()
    plt.rcParams['figure.figsize'] = [30, 15]
    y_hat_heatmap = V_C.copy()
    for i in range(y_s.shape[1]):
        y_hat_heatmap[i][i] = (np.sum(y_hat_heatmap[i])/(len(y_hat_heatmap[i]) - 1))
        sns.heatmap((y_hat_heatmap-np.mean(y_hat_heatmap))/(np.std(y_hat_heatmap)), center=0, annot=True, fmt="f")

def computing_V_C(Y_hat, show_heat=False):
    V_C = np.zeros((Y_hat.shape[0], Y_hat.shape[0]))
    dtype = torch.FloatTensor
    for i in range(Y_hat.shape[0]):
        y_hat_i = torch.tensor(Y_hat[i,:]).type(dtype)
        for j in range(i + 1, Y_hat.shape[0]):
            y_hat_j =  torch.tensor(Y_hat[i,:]).type(dtype)
            V_C[i][j] = V_C[j][i] = Vc_distance(y_hat_i, y_hat_j)

    if show_heat:
        show_normalized_heatmap(V_C)

    for i in range(y_s.shape[1]):
        y_hat_i = torch.tensor(Y_hat[:, i]).type(dtype)
        V_C[i][i] = Vc_distance(y_hat_i, y_hat_i)

    V_C = torch.from_numpy(V_C).float()
    return V_C

def computing_D_C(V_C, y_s):
    D_C = []
    for i in range(y_s.shape[1]):
        D_C.append(sum(V_C[i]))
    D_C = torch.Tensor(D_C).float()
    D_C = torch.diag(D_C)
    return D_C

def computing_L_C(D_C, V_C):
    D_C_sqrt_inverse = torch.sqrt(torch.inverse(D_C))
    L_C = torch.eye(18) - torch.mm(D_C_sqrt_inverse, torch.mm(V_C, D_C_sqrt_inverse))
    return L_C

def trace_z_class_and_samples(L_C, V_X, V_C, Z):
    trace_z_class_z = torch.trace(torch.mm(torch.mm(Z.permute((1,0)), L_C), Z))
    trace_z_samples_z = torch.trace(torch.mm(torch.mm(Z, L_X), Z.permute((1,0))))
    return trace_z_class_z, trace_z_samples_z

def l_bar_class_and_samples(D_X, D_C, V_X, V_C):
    D_X_sqrt_inverse = torch.sqrt(torch.inverse(D_X))
    D_C_sqrt_inverse = torch.sqrt(torch.inverse(D_C))
    L_bar_X = torch.mm(D_X_sqrt_inverse, torch.mm(V_X, D_X_sqrt_inverse))
    L_bar_C = torch.mm(D_C_sqrt_inverse, torch.mm(V_C, D_C_sqrt_inverse))
    return L_bar_C, L_bar_X

# Data preparation
def train_test_split(split_ratio=.80, drop_rate=0.4):
    # Reading file
    data_df = pd.read_csv('yeast.data', header=None, delimiter=r"\s+")
    mlb = MultiLabelBinarizer()
    print(data_df.columns.values)
    data_df = data_df.join(pd.DataFrame(mlb.fit_transform(data_df.pop(9)),
                              columns=mlb.classes_,
                              index=data_df.index))
    y_modified_data_df = data_df.copy()
    # Dropping random cells per sample
    y_modified_data_df.iloc[:,-1 * len(mlb.classes_):] = y_modified_data_df.iloc[:, -1 * len(mlb.classes_):].replace(0, -1)
    dropped_y_modified_data_df = dataset_drop_labels(y_modified_data_df, 9, len(mlb.classes_), drop_rate)
    x_s, y_s = dropped_y_modified_data_df.iloc[:,1:9], dropped_y_modified_data_df.iloc[:,9:]
    # Sanity Check
    print(dropped_y_modified_data_df.shape[0] == len((y_s.T != 0).any()))
    x_s, y_s = x_s.to_numpy(), y_s.to_numpy()

    # Splitting by train and test set
    random_indices = list(range(x_s.shape[0]))
    random.shuffle(random_indices)
    split_ratio = int(len(random_indices) * .80)
    train_items, test_items = random_indices[:split_ratio], random_indices[split_ratio:]
    train_set = (x_s[train_items, :], y_s[train_items, :])
    test_set = (x_s[test_items, :], y_s[test_items, :])
    return train_set, test_set

train_set, test_set = train_test_split()

# Example, just using train
x_s, y_s = train_set
Y = Y_hat = y_s.T

# Computing all steps example
Z = torch.randn(*Y.shape)
V_X = computing_V_X(x_s)
D_X = computing_D_X(V_X)
L_X = computing_L_X(V_X, D_X)
V_C = computing_V_C(Y_hat)
D_C = computing_D_C(V_C, y_s)
L_C = computing_L_C(D_C, V_C)

trace_z_class, trace_z_sample = trace_z_class_and_samples(L_C, V_X, V_C, Z)
L_bar_C, L_bar_X = l_bar_class_and_samples(D_X, D_C, V_X, V_C)

# Hyperparameters
lambda_X, lambda_C = [1.0] * 2
a_x = lambda_X / (lambda_X + 1)
a_c = lambda_C / (lambda_C + 1)

# Solving
Y_tens = torch.tensor(Y).float()
class_solve = torch.mm(torch.inverse(torch.eye(D_C.shape[0]) - a_c * L_bar_C), Y_tens)
mlml_appro = torch.mm(class_solve, (torch.inverse(torch.eye(D_X.shape[0]) - a_x * L_bar_X)))
