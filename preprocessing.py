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



def pdb2pandas(pdb_path):
    df = PandasPdb().read_pdb(pdb_path).df["ATOM"]
    df["node_id"] = (
        df["chain_id"]
        + df["residue_number"].map(str)
        + df["residue_name"]
    )
    df["residue_id"] = df["residue_name"]
    mapping = {'DA': 0, 'A': 0, 'C': 1,'DC': 1, 'DG': 2, 'G': 2, 'U': 3, 'DU': 3, 'DT': 3, 'I':4, 'DI':4}
    df = df.replace({"residue_id":mapping})
    return df


def normalize_pc(points):
	centroid = np.mean(points, axis=0)
	points -= centroid
	furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
	points /= furthest_distance
	return points



def create_pyg_datalist(data_path, max_pointcloud_size, withfeatures=True):
    datalist = []
    shape_list = []
    for filename in track(os.listdir(data_path), description="[cyan]Creating PyG Data from RNA pdb files"):
    # for filename in tqdm(os.listdir(data_path), desc="Parsing Data from RNA pdb files"):
        if filename.endswith("pdb"):
            pdb_df = pdb2pandas(os.path.join(data_path, filename))
            coordinates = pdb_df[["x_coord","y_coord","z_coord"]].to_numpy()         # should be shape (num_atoms,3)
            coordinates = normalize_pc(coordinates)
            atom_number = torch.from_numpy(pdb_df[["atom_number"]].to_numpy())
            residue_ids = np.eye(5)[pdb_df["residue_id"].to_numpy().astype(float).astype(int)]
            coordinates = np.concatenate((coordinates, residue_ids), axis=1) if withfeatures else coordinates
            node_id = pdb_df[["node_id"]].to_numpy()
            shape_list.append(coordinates.shape[0])
            if coordinates.shape[0] <= max_pointcloud_size and coordinates.shape[0] >=100:
                paddingamount = max_pointcloud_size - coordinates.shape[0]
                coords_padded = torch.from_numpy(np.pad(coordinates, ((0, paddingamount),(0, 0)), 'constant', constant_values=(0, 0))).type(torch.FloatTensor)    
                data = Data(pos=coords_padded, atom_number=atom_number, y=node_id, num_nodes=coords_padded.shape[0])
                datalist.append(data)
            # else:
            #     raise Exception("Incorrect max_pointcloud_size")
            
    return datalist, shape_list


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