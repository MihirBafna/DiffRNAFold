import numpy as np
import pandas as pd
import networkx as nx
import os
import argparse
from rich.table import Table
from rich.console import Console
import torch
import preprocessing
import models
import wandb
import training
from pytorch3d.loss import chamfer_distance
# from chamferdist import ChamferDistance


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
    default_config = {
        "lr": 0.000790669,
        "latent_dim": 512,
        "model_type": "AE",
        "augmented": False
    }
    args = parse_arguments()
    if args is None or True:
        mode = "trainAE"
        data_dir_path = r'./data/raw/all_representative_pdb_4_0__3_258/'
        output_dir_path = r'./out/'
        studyname = "PDB(100-140)_Normalized_AE_Metadata"
    else:
        mode = args.mode
        data_dir_path = args.inputdirpath
        output_dir_path = args.outputdirpath
        studyname = args.studyname
    
    preprocess_output_path = os.path.join(output_dir_path, "preprocessed/")
    training_output_path = os.path.join(output_dir_path, "train/")
    # modularize hyperparameter selection
    epochs = 1000
    num_features = 0
    intermediate_save_path = None
    batchSize= 16
    pointcloudsize = 140
    with_features=False
    withval = True
    augment_num = 1 # no augmentation

    criterion = chamfer_distance
    
    if "preprocess" in mode:
    
        print("\n#------------------------------ Preprocessing ----------------------------#\n")

        data_list, _ = preprocessing.create_pytorch_datalist(data_dir_path, pointcloudsize, withfeatures=with_features, augment_num=augment_num)
        train_loader, test_loader, val_loader = preprocessing.create_dataloaders(data_list, batch_size=batchSize, with_val=withval)
        
        if not os.path.exists(preprocess_output_path):
            os.mkdir(preprocess_output_path)
        torch.save(train_loader, os.path.join(preprocess_output_path,f'train_dataloader_{studyname}.pth'))
        torch.save(test_loader, os.path.join(preprocess_output_path,f'test_dataloader_{studyname}.pth'))
        if withval:
            torch.save(val_loader, os.path.join(preprocess_output_path,f'val_dataloader_{studyname}.pth'))

    if "trainAE" in mode:
        print("\n#------------------------------ 1. Training Point Cloud Autoencoder ----------------------------#\n")
        wandb.init(project="DiffFold-Sweep", entity="diffrnafold", config=default_config)
        if not "preprocess" in mode:
            if wandb.config.augmented:
                studyname = "PDB(100-140)_Normalized_AE_Augmented_Metadata"
            train_loader = torch.load(os.path.join(preprocess_output_path,f'train_dataloader_{studyname}.pth'))
            test_loader = torch.load(os.path.join(preprocess_output_path,f'test_dataloader_{studyname}.pth'))
            if withval:
                val_loader = torch.load(os.path.join(preprocess_output_path,f'val_dataloader_{studyname}.pth'))


        if epochs >= 500:
            intermediate_save_path1 = os.path.join(training_output_path,f'trained_model_{studyname}_100epochs.pth')
            intermediate_save_path2 = os.path.join(training_output_path,f'trained_model_{studyname}_500epochs.pth')
            intermediate_save_path3 = os.path.join(training_output_path,f'trained_model_{studyname}_checkpointepochs.pth')
            # intermediate_save_path = (intermediate_save_path1, intermediate_save_path2, intermediate_save_path3)
            
        if not os.path.exists(training_output_path):
            os.mkdir(training_output_path)
        if wandb.config.model_type == "VAE":
            model = models.PointVAE(num_points=pointcloudsize, latent_dim=wandb.config.latent_dim, num_features=num_features)
        elif wandb.config.model_type == "AE":
            model = models.PointAutoEncoder(num_points=pointcloudsize, latent_dim=wandb.config.latent_dim, num_features=num_features)
        else:
            model = models.PointAutoEncoderV2(num_points=pointcloudsize, latent_dim=wandb.config.latent_dim, num_features=num_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
        trained_model = training.train_model(model, optimizer, train_loader, epochs, criterion, val_loader, batchSize, intermediate_save_path)

        torch.save(trained_model.state_dict(), os.path.join(training_output_path,f'trained_model_{studyname}_{epochs}epochs.pth'))

    if "trainDiff" in mode:

        print("\n#------------------------------ 2. Training Diffusion Model ----------------------------#\n")
        if epochs >= 500:
            intermediate_save_path1 = os.path.join(training_output_path,f'trained_model_diffusion_{studyname}_100epochs.pth')
            intermediate_save_path2 = os.path.join(training_output_path,f'trained_model_diffusion_{studyname}_500epochs.pth')
            intermediate_save_path3 = os.path.join(training_output_path,f'trained_model_diffusion_{studyname}_checkpointepochs.pth')
            # intermediate_save_path = (intermediate_save_path1, intermediate_save_path2, intermediate_save_path3)
        wandb.init(project="DiffRNAFold-Diffusion", entity="diffrnafold", config=default_config)
        if not "preprocess" in mode:
            train_loader = torch.load(os.path.join(preprocess_output_path,f'train_dataloader_{studyname}.pth'))
            test_loader = torch.load(os.path.join(preprocess_output_path,f'test_dataloader_{studyname}.pth'))
            if withval:
                val_loader = torch.load(os.path.join(preprocess_output_path,f'val_dataloader_{studyname}.pth'))

        if not "trainAE" in mode:
            if wandb.config.model_type == "VAE":
                pointcloud_ae = models.PointVAE(num_points=pointcloudsize, latent_dim=512, num_features=num_features)
            elif wandb.config.model_type == "AE":
                pointcloud_ae = models.PointAutoEncoder(num_points=pointcloudsize, latent_dim=512, num_features=num_features)
            else:
                pointcloud_ae = models.PointAutoEncoderV2(num_points=pointcloudsize, latent_dim=512, num_features=num_features)

            pointcloud_ae.load_state_dict(torch.load(os.path.join(training_output_path,f'trained_model_PDB(100-140)_Normalized_AE_Metadata_1000epochs.pth')))

        assert pointcloud_ae is not None

        for para in pointcloud_ae.parameters():
            para.requires_grad = False
        
        latentdiffusionModel = models.LatentSpaceDiffusion1D(pointcloudsize, wandb.config.latent_dim, pointcloud_ae)

        optimizer = torch.optim.Adam(latentdiffusionModel.parameters(), lr=wandb.config.lr)
        trained_diffusion_model = training.train_model(latentdiffusionModel, optimizer, train_loader, epochs, criterion, val_loader, batchSize, intermediate_save_path)
        torch.save(trained_diffusion_model.state_dict(), os.path.join(training_output_path,f'trained_diffusion_model{studyname}_{epochs}epochs.pth'))





if __name__ == "__main__":
    wandb.agent("8c04amnx", project="DiffFold-Sweep", entity="diffrnafold", function=main)
    # main()