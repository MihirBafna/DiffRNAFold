import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
from rich.progress import track
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from biopandas.pdb import PandasPdb
from collections import defaultdict
import rdkit.Chem as Chem
from featurizers import PDBFeaturizer
import torch_geometric.transforms as T

def pdb2pandas(pdb_path):
    df = PandasPdb().read_pdb(pdb_path).df["ATOM"]
    df["node_id"] = (
        df["chain_id"]
        + df["residue_number"].map(str)
        + df["residue_name"]
    )
    residue_mapping = defaultdict(lambda: 8)
    residue_mapping["A"] = 0
    residue_mapping["T"] = 1
    residue_mapping["U"] = 1
    residue_mapping["C"] = 2
    residue_mapping["G"] = 3
    residue_mapping["DA"] = 4
    residue_mapping["DT"] = 5
    residue_mapping["DU"] = 5
    residue_mapping["DC"] = 6
    residue_mapping["DG"] = 7

    element_mapping = {'C':0, 'N':1, 'O':2, 'P':3}

    df["residue_id"] = df["residue_name"].map(residue_mapping)
    df["element_id"] = df["element_symbol"].map(element_mapping)

    return df


def augment_pc(points):
    
    theta_x, theta_y, theta_z = tuple(np.random.rand(3) * 2 *np.pi)
    
    
    rotate_x = np.array([[1,0,0],
                         [0,  np.cos(theta_x), -np.sin(theta_x)],
                         [0, np.sin(theta_x), np.cos(theta_x)]])
    rotate_y = np.array([[np.cos(theta_y), 0,  -np.sin(theta_y)],
                         [0,1,0],
                         [np.sin(theta_y), 0 , np.cos(theta_y)]])  
    rotate_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                         [np.sin(theta_z), np.cos(theta_z), 0],
                         [0,0, 1]])
    return  points @ (rotate_z @ rotate_y @ rotate_x).T


def pad_pc(points, amount):
    coords_padded = np.pad(points, ((0, amount),(0, 0)), 'constant', constant_values=(0, 0))
    return coords_padded

def get_maximum_distance(file_path):
    points = PandasPdb().read_pdb(file_path).df["ATOM"][["x_coord", "y_coord", "z_coord"]].values
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    return furthest_distance

def process_dfs(data_path):
    dists = []
    for filename in track(os.listdir(data_path), description="[cyan]Processing maximum distance from PDB files"):
        if filename.endswith("pdb"):
            file_path = os.path.join(data_path, filename)
            dists.append(get_maximum_distance(file_path))
    return max(dists)

featurizer = PDBFeaturizer(use_edges=True)
def process_file(data_path, max_nodes, filename, furthest_distance):
    if filename.endswith("pdb"):
        file_path = os.path.join(data_path, filename)
        return featurizer.featurize(file_path, max_nodes, furthest_distance=furthest_distance)
    return None, False

    

def create_pytorch_datalist(data_path, max_nodes, withfeatures=True, augment_num=10):
    pad = T.Pad(max_num_nodes=max_nodes)
    datalist = []
    shape_list = []
    # furthest_distance = process_dfs(data_path)
    furthest_distance = 14
    for filename in track(os.listdir(data_path), description="[cyan]Creating PyG Data from RNA pdb files"):
        graph, is_valid = process_file(data_path, max_nodes, filename, furthest_distance)
        if(is_valid):
            datalist.append(pad(graph.to_pyg_graph()))
        
    return datalist



# def create_pytorch_datalist(data_path, max_pointcloud_size, withfeatures=True, augment_num=10):
#     datalist = []
#     shape_list = []
#     for filename in track(os.listdir(data_path), description="[cyan]Creating PyG Data from RNA pdb files"):
#         if filename.endswith("pdb"):
#             pdb_df = pdb2pandas(os.path.join(data_path, filename))
            
#             atom_number = torch.from_numpy(pdb_df[["atom_number"]].to_numpy())
#             residue_ids = pdb_df['residue_id'].to_numpy()
#             element_ids = pdb_df['element_id'].to_numpy()
#             node_id = pdb_df[["node_id"]].to_numpy()
            
#             raw_coordinates = pdb_df[["x_coord","y_coord","z_coord"]].to_numpy()         # should be shape (num_atoms,3)
            
#             if raw_coordinates.shape[0] <= max_pointcloud_size and raw_coordinates.shape[0] >=100:
                
#                 paddingamount = max_pointcloud_size - raw_coordinates.shape[0]
#                 shape_list.append(raw_coordinates.shape[0])

#                 normalized_coordinates = normalize_pc(raw_coordinates)
                
#                 temp_coordinates = normalized_coordinates.copy()    
#                 for i in range(augment_num):
#                     feature_coordinates = np.concatenate((temp_coordinates, residue_ids, element_ids), axis=1) if withfeatures else temp_coordinates
#                     padded_coordinates = pad_pc(feature_coordinates, paddingamount)
#                     # data = Data(pos=torch.from_numpy(padded_coordinates).type(torch.FloatTensor), atom_number=atom_number, y=node_id, num_nodes=padded_coordinates.shape[0])
#                     # print(data)
#                     # data = {"pos": padded_coordinates.type(torch.FloatTensor), "atom_number":atom_number, "y": node_id, "num_nodes": padded_coordinates.shape[0]}
#                     data = {"x": torch.from_numpy(padded_coordinates).type(torch.FloatTensor), "metadata": filename,"atom_number":atom_number, "y": node_id, "num_nodes": padded_coordinates.shape[0]}
#                     datalist.append(data)
#                     temp_coordinates = augment_pc(normalized_coordinates.copy())

#     return datalist, shape_list


# class PDBDataset(Dataset):
#     def __init__(self, data):
#         self.pdbs = data
#     def __len__(self):
#         return len(self.pdbs)
#     def __getitem__(self, idx):
#         pdb = self.pdbs[idx]
#         return pdb["x"], pdb["metadata"]
    
    
def create_dataloaders(data_list, batch_size=1, with_val= False):
    X_train, X_t = train_test_split(data_list, test_size=0.2, random_state=42)
    
    if with_val:
        X_val, X_test = train_test_split(X_t, test_size=0.5, random_state=42)

        assert len(X_train) + len(X_val) + len(X_test) == len(data_list)
        
        train_data_loader = DataLoader(X_train, batch_size=batch_size)
        val_data_loader = DataLoader(X_val, batch_size=batch_size)
        test_data_loader = DataLoader(X_test, batch_size=batch_size)
        
        return train_data_loader, test_data_loader, val_data_loader
    else:
        assert len(X_train) + len(X_t)== len(data_list)
        train_data_loader = DataLoader(X_train, batch_size=batch_size)
        test_data_loader = DataLoader(X_t, batch_size=batch_size)
        
        return train_data_loader, test_data_loader, None