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
    return df


def create_pyg_datalist(data_path, max_pointcloud_size=1233):
    datalist = []
    shape_list = []
    for filename in track(os.listdir(data_path), description="[cyan]Creating PyG Data from RNA pdb files"):
    # for filename in tqdm(os.listdir(data_path), desc="Parsing Data from RNA pdb files"):
        if filename.endswith("pdb"):
            pdb_df = pdb2pandas(os.path.join(data_path, filename))
            coordinates = pdb_df[["x_coord","y_coord","z_coord"]].to_numpy()         # should be shape (num_atoms,3)
            atom_number = torch.from_numpy(pdb_df[["atom_number"]].to_numpy())
            node_id = pdb_df[["node_id"]].to_numpy()
            shape_list.append(coordinates.shape[0])
            if coordinates.shape[0] <= max_pointcloud_size:
                paddingamount = max_pointcloud_size - coordinates.shape[0]
                coords_padded = torch.from_numpy(np.pad(coordinates, ((0, paddingamount),(0, 0)), 'constant', constant_values=(0, 0))).type(torch.FloatTensor)    
                data = Data(pos=coords_padded, atom_number=atom_number, y=node_id, num_nodes=coords_padded.shape[0])
                datalist.append(data)
            # else:
            #     raise Exception("Incorrect max_pointcloud_size")
            
    return datalist, shape_list


def create_dataloaders(data_list, batch_size=1):
    X_train, X_t = train_test_split(data_list, test_size=0.33, random_state=42)
    X_val, X_test = train_test_split(X_t, test_size=0.5, random_state=42)

    assert len(X_train) + len(X_val) + len(X_test) == len(data_list)
    
    train_data_loader = DataLoader(X_train, batch_size=batch_size)
    val_data_loader = DataLoader(X_val, batch_size=batch_size)
    test_data_loader = DataLoader(X_test, batch_size=batch_size)
    
    return train_data_loader, val_data_loader, test_data_loader 