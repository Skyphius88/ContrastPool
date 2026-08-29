# # -*- coding: utf-8 -*-
# import pandas as pd
# import numpy as np
# import networkx as nx
# import os  # To create directories
# import shutil
# import scipy.io
# import dgl
# import torch
# import glob
# import csv
# import re
# import json
# from tqdm import tqdm
# from dgl.data.utils import save_graphs
# from sklearn.model_selection import StratifiedKFold, train_test_split


# def _load_matrix_subject_with_files(files, remove_negative=False):
#     subjects = []
#     for file in files:
#         mat = scipy.io.loadmat(file)
#         mat = mat["data"]
#         np.fill_diagonal(mat, 0)
#         if remove_negative:
#             mat[mat < 0] = 0
#         subjects.append(mat)
#     return np.array(subjects)

# def construct_dataset(data_name):
#     feat_dir = 'data/to/connectivity_matrices_schaefer/' + data_name + '/'

#     G_dataset = []
#     Labels = []
#     group2idx = {}
#     paths = glob.glob(feat_dir + '/*/' + '*_features_timeseries.mat', recursive=True)
#     feats = _load_matrix_subject_with_files(paths)

#     print('Processing ' + data_name + '...')

#     for j in tqdm(range(len(feats))):
#         name = paths[j].split('/')[-1]
#         group = re.findall('sub-([^\d]+)', name)[0]
#         if group not in group2idx.keys():
#             group2idx[group] = len(group2idx.keys())
#         i = group2idx[group]

#         G = nx.DiGraph(np.ones([feats[j].shape[0], feats[j].shape[0]]))
#         graph_dgl = dgl.from_networkx(G)

#         graph_dgl.ndata['N_features'] = torch.from_numpy(feats[j])
#         # Include edge features
#         weights = []
#         for u, v, w in G.edges.data('weight'):
#             # if w is not None:
#             weights.append(w)
#         graph_dgl.edata['E_features'] = torch.Tensor(weights)

#         G_dataset.append(graph_dgl)
#         Labels.append(i)

#     print('Finish process ' + data_name + '. ' + str(len(feats)) + ' subjects in total.')

#     Labels = torch.LongTensor(Labels)
#     graph_labels = {"glabel": Labels}
#     if not os.path.exists('./bin_dataset/'):
#         os.mkdir('./bin_dataset/')
#     print(Labels.shape)
#     print(len(G_dataset))
#     save_graphs("./bin_dataset/" + data_name + ".bin", G_dataset, graph_labels)


# def move_files(data_name):
#     feat_dir = '/data/jiaxing/brain/connectivity_matrices_schaefer/' + data_name + '/'
#     paths = glob.glob(feat_dir + '/*/*', recursive=True)
#     for path in paths:
#         if path[-4:] == '.mat':
#             if 'schashaefer' in path:
#                 new_path = re.sub('schashaefer', 'schaefer', path)
#                 os.rename(path, new_path)
#             continue
#         else:
#             parcellation = data_name.split('_')[-1]
#             os.rename(path, path + '_' + parcellation + '_correlation_matrix.mat')


# if __name__ == '__main__':
#     error_name = []
#     # file_name_list = os.listdir('./correlation_datasets/')
#     file_name_list = ['adni_schaefer100']

#     for data_name in file_name_list:
#         move_files(data_name)
#         # construct_dataset(data_name)
#         # try:
#         #     construct_dataset(data_name)
#         # except:
#         #     print('[ERROR]: ' + data_name)
#         #     error_name.append(data_name)
#     print(error_name)
# #     print('Done!')



    # -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
import os
import scipy.io
import dgl
import torch
import glob
import re
from tqdm import tqdm
from dgl.data.utils import save_graphs


def _load_matrix_subject_with_files(files, expected_nodes=100, remove_negative=False):
    valid_subjects = []
    valid_paths = []
    
    for file in files:
        try:
            mat = scipy.io.loadmat(file)
            key = 'data' if 'data' in mat else [k for k in mat.keys() if not k.startswith('__')][0]
            matrix = np.array(mat[key])
            
            # Ensure matrix is 2D and matches expected node dimensions (100x100)
            if matrix.ndim == 2 and matrix.shape[0] == expected_nodes and matrix.shape[1] == expected_nodes:
                np.fill_diagonal(matrix, 0)
                if remove_negative:
                    matrix[matrix < 0] = 0
                valid_subjects.append(matrix)
                valid_paths.append(file)
        except Exception:
            continue
            
    return np.array(valid_subjects), valid_paths


def construct_dataset(data_name):
    num_nodes_match = re.search(r'\d+', data_name)
    expected_nodes = int(num_nodes_match.group(0)) if num_nodes_match else 100

    # Extract base name (e.g. 'neurocon' from 'neurocon_schaefer100')
    prefix = data_name.split('_')[0]

    # Search paths mapped to your workspace layout
    search_dirs = [
        f'./raw_data/{prefix}',
        f'./raw_data/{data_name}',
        f'./data/{data_name}',
        f'./{prefix}',
        f'./{data_name}'
    ]
    
    paths = []
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            found = glob.glob(os.path.join(s_dir, '**', '*.mat'), recursive=True)
            if found:
                schaefer_files = [f for f in found if 'schaefer100' in f.lower() or '100' in f]
                paths = schaefer_files if schaefer_files else found
                print(f"[{data_name}] Found {len(paths)} potential .mat files in: {s_dir}")
                break

    if not paths:
        print(f"[ERROR] Could not find any .mat files for {data_name} in {search_dirs}!")
        return

    print(f"Filtering and loading {expected_nodes}x{expected_nodes} matrices for {data_name}...")
    feats, valid_paths = _load_matrix_subject_with_files(paths, expected_nodes=expected_nodes)

    if len(feats) == 0:
        print(f"[ERROR] No valid {expected_nodes}x{expected_nodes} matrices found for {data_name}!")
        return

    G_dataset = []
    Labels = []
    group2idx = {}

    for j in tqdm(range(len(feats))):
        name = os.path.basename(valid_paths[j])
        
        # Extract class label from filename ('sub-control', 'sub-patient') or parent directory
        group_match = re.findall(r'sub-([^\d_]+)', name)
        if not group_match:
            parent_dir = os.path.basename(os.path.dirname(valid_paths[j]))
            group = parent_dir if parent_dir not in [prefix, 'raw_data'] else 'default'
        else:
            group = group_match[0]
        
        if group not in group2idx:
            group2idx[group] = len(group2idx)
        label_idx = group2idx[group]

        num_nodes = feats[j].shape[0]
        G = nx.DiGraph(np.ones([num_nodes, num_nodes]))
        graph_dgl = dgl.from_networkx(G)

        graph_dgl.ndata['N_features'] = torch.from_numpy(feats[j]).float()
        
        weights = [w for u, v, w in G.edges.data('weight')]
        graph_dgl.edata['E_features'] = torch.Tensor(weights).float()

        G_dataset.append(graph_dgl)
        Labels.append(label_idx)

    print(f'Finished processing {data_name}. {len(feats)} valid subjects compiled. Classes detected: {group2idx}')

    Labels = torch.LongTensor(Labels)
    graph_labels = {"glabel": Labels}
    
    os.makedirs('./data', exist_ok=True)
    out_path = f"./data/{data_name}.bin"
    save_graphs(out_path, G_dataset, graph_labels)
    print(f"Successfully generated binary graph: {out_path}\n")


if __name__ == '__main__':
    # Add any additional datasets from your raw_data directory here
    file_name_list = [
        #'neurocon_schaefer100',
        # 'adni_schaefer100',
        #'ppmi_schaefer100',
         'taowu_schaefer100',
    ]

    for data_name in file_name_list:
        construct_dataset(data_name)
    print('Processing complete!')