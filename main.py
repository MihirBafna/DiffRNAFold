import numpy as np
import pandas as pd
import networkx as nx
import os
import argparse
from rich.table import Table
from rich.console import Console
import torch
from chamferdist import ChamferDistance
import preprocessing
import models
import wandb
import training



def parse_arguments():
    parser = argparse.ArgumentParser(description='clarifyGAE arguments')
    parser.add_argument("-m", "--mode", type=str, default = "train",
        help="Pipeline Mode: preprocess,train,test")
    parser.add_argument("-i", "--inputdirpath", type=str,
                    help="Input directory path where PDB data is stored")
    parser.add_argument("-o", "--outputdirpath", type=str,
                    help="Output directory path where results will be stored ")
    parser.add_argument("-s", "--studyname", type=str,
                help="Name of study")
    args = parser.parse_args()
    return args




def main():
    args = parse_arguments()
    mode = args.mode
    data_dir_path = args.inputdirpath
    output_dir_path = args.outputdirpath
    studyname = args.studyname
    
    preprocess_output_path = os.path.join(output_dir_path, "preprocessed")
    training_output_path = os.path.join(output_dir_path, "train")

    pointcloudsize = 140
    withval = True
    
    if "preprocess" in mode:
    
        print("\n#------------------------------ Preprocessing ----------------------------#\n")

        data_list, _ = preprocessing.create_pyg_datalist(data_dir_path, pointcloudsize, withfeatures=False)
        train_loader, test_loader, val_loader = preprocessing.create_dataloaders(data_list, batch_size=1, with_val=withval)
        
        if not os.path.exists(preprocess_output_path):
            os.mkdir(preprocess_output_path)
        torch.save(train_loader, os.path.join(preprocess_output_path,f'train_dataloader_{studyname}.pth'))
        torch.save(test_loader, os.path.join(preprocess_output_path,f'test_dataloader_{studyname}.pth'))
        if withval:
            torch.save(val_loader, os.path.join(preprocess_output_path,f'val_dataloader_{studyname}.pth'))

    if "train" in mode:
        print("\n#------------------------------ 1. Training Point Cloud Autoencoder ----------------------------#\n")

        if not "preprocess" in mode:
            train_loader = torch.load(os.path.join(preprocess_output_path,f'train_dataloader_{studyname}.pth'))
            test_loader = torch.load(os.path.join(preprocess_output_path,f'test_dataloader_{studyname}.pth'))
            if withval:
                val_loader = torch.load(os.path.join(preprocess_output_path,f'val_dataloader_{studyname}.pth'))

        # modularize hyperparameter selection
        epochs = 1000
        num_features = 0

        model = models.PointAutoEncoder(num_points=pointcloudsize, latent_dim=128, num_features=num_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = ChamferDistance()
        intermediate_save_path = None

        if epochs >= 500:
            intermediate_save_path1 = os.path.join(training_output_path,f'trained_model_{studyname}_100epochs.pth')
            intermediate_save_path2 = os.path.join(training_output_path,f'trained_model_{studyname}_500epochs.pth')
            intermediate_save_path = (intermediate_save_path1, intermediate_save_path2)
            
        wandb.init(project="DiffRNAFold", entity="diffrnafold")

        if not os.path.exists(training_output_path):
            os.mkdir(training_output_path)

        trained_model = training.train_model(model, optimizer, train_loader, epochs, criterion, val_loader, intermediate_save_path)

        torch.save(trained_model.state_dict(), os.path.join(training_output_path,f'trained_model_{studyname}_{epochs}epochs.pth'))

if __name__ == "__main__":
    main()