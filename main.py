import numpy as np
import pandas as pd
import networkx as nx
import os
import argparse
from rich.table import Table
from rich.console import Console
import torch
import preprocessing
import wandb
os.environ["WANDB__SERVICE_WAIT"] = "300"

# from pytorch3d.loss import chamfer_distance
import lightning.pytorch as pl



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


def build_gae_model(hyperparams, state_dict=None):    

    from models import gnns
    from models.LightningGAE import LightningGAE
    
    # in_dim = hyperparams["max_nodes"]
    in_dim = 33
    embed_dim = hyperparams["embed_dim"]
    edge_dim = hyperparams["edge_dim"]
    coord_dim = 3
    
    encoder = gnns.GraphEncoder(in_dim, embed_dim, edge_dim )
    decoder = gnns.GraphDecoder(embed_dim, coord_dim)
    gae = LightningGAE(encoder,decoder)
    
    if state_dict is not None:
        gae.load_state_dict(torch.load(state_dict))

    return gae


def build_diffusion_model(hyperparams, state_dict=None):
    
    from models import diffusion
    
    iteration_architecture = diffusion.Unet1D(
        dim = hyperparams["embed_dim"],
        dim_mults = (1, 2, 4, 8),
        channels= hyperparams["num_nodes"]
    )
        
    diffusion = diffusion.GaussianDiffusion1D(
        iteration_architecture,
        seq_length = hyperparams["embed_dim"],
        timesteps = 1000,
        objective = 'pred_x0'
    )
    
    return diffusion


def main():
    default_config = {
        "latent_dim": 512,
        "model_type": "GAE"
    }
    args = parse_arguments()
    if args is None or True:
        mode = "train_diffusion"
        data_dir_path = r'./data/raw/all_representative_pdb_4_0__3_258'
        output_dir_path = r'./out'
        studyname = "GraphBasedAtomBondEncoded_nobatch"
    else:
        mode = args.mode
        data_dir_path = args.inputdirpath
        output_dir_path = args.outputdirpath
        studyname = args.studyname
    
    preprocess_output_path = os.path.join(output_dir_path, "preprocessed")
    training_output_path = os.path.join(output_dir_path, "train")
    # modularize hyperparameter selection
    epochs = 400
    num_features = 0
    intermediate_save_path = None
    # batchSize= 16
    batchSize = 1
    pointcloudsize = 140
    augment_num=10
    with_features=False
    withval = True
    augment_num = 1 # no augmentation
    criterion = None
    
    edge_dim = 11
    
    hyperparameters = {
        "num_nodes":pointcloudsize,
        "in_dim": 33,
        "embed_dim":512,
        "edge_dim": 11
    }
    
    if "preprocess" in mode:
    
        print("\n#------------------------------ 1. Preprocessing ----------------------------#\n")

        data_list = preprocessing.create_pytorch_datalist(data_dir_path, pointcloudsize, withfeatures=with_features, augment_num=augment_num)
        train_loader, test_loader, val_loader = preprocessing.create_dataloaders(data_list, batch_size=batchSize, with_val=withval)
        
        if not os.path.exists(preprocess_output_path):
            os.mkdir(preprocess_output_path)
        torch.save(train_loader, os.path.join(preprocess_output_path,f'train_dataloader_{studyname}.pth'))
        torch.save(test_loader, os.path.join(preprocess_output_path,f'test_dataloader_{studyname}.pth'))
        if withval:
            torch.save(val_loader, os.path.join(preprocess_output_path,f'val_dataloader_{studyname}.pth'))

    if "train_gae" in mode:
        print("\n#------------------------------ 2. Training Autoencoder ----------------------------#\n")

        if not "preprocess" in mode:
            train_loader = torch.load(os.path.join(preprocess_output_path,f'train_dataloader_{studyname}.pth'))
            test_loader = torch.load(os.path.join(preprocess_output_path,f'test_dataloader_{studyname}.pth'))
            if withval:
                val_loader = torch.load(os.path.join(preprocess_output_path,f'val_dataloader_{studyname}.pth'))


        if epochs >= 500:
            intermediate_save_path1 = os.path.join(training_output_path,f'trained_model_{studyname}_100epochs.pth')
            intermediate_save_path2 = os.path.join(training_output_path,f'trained_model_{studyname}_500epochs.pth')
            intermediate_save_path3 = os.path.join(training_output_path,f'trained_model_{studyname}_checkpointepochs.pth')
            intermediate_save_path = (intermediate_save_path1, intermediate_save_path2, intermediate_save_path3)
            
        # wandb.init(project="DiffFold-Sweep", entity="diffrnafold", mode="disabled", config=default_config)
        if not os.path.exists(training_output_path):
            os.mkdir(training_output_path)
            
        # if wandb.config.model_type == "VAE":
        #     model = models.PointVAE(num_points=pointcloudsize, latent_dim=wandb.config.latent_dim, num_features=num_features)
        # elif wandb.config.model_type == "AE":
        #     model = models.PointAutoEncoder(num_points=pointcloudsize, latent_dim=wandb.config.latent_dim, num_features=num_features)
        #     optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
        #     trained_model = training.train_model(model, optimizer, train_loader, epochs, criterion, val_loader, batchSize, intermediate_save_path)
        #     torch.save(trained_model.state_dict(), os.path.join(training_output_path,f'trained_model_{studyname}_{epochs}epochs.pth'))


        # if  wandb.config.model_type == "GAE":       # Graph based 
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=epochs) # add hyperparams here 
        # hyperparameters = dict()
        hyperparameters = {
        "max_nodes":pointcloudsize,
        "embed_dim":default_config["latent_dim"],
        "edge_dim":edge_dim
        }
        model = build_gae_model(hyperparameters, type="GAE")
        wandb.init(project="DiffRNAFold", entity="diffrnafold",mode="online")
        trainer.fit(model, train_loader, val_loader)
            
        torch.save(model.state_dict(), os.path.join(training_output_path,f'trained_model_{studyname}_{epochs}epochs.pth'))
    
    if "train_diffusion" in mode:
        print("\n#------------------------------ 3. Training Diffusion ----------------------------#\n")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not "preprocess" in mode:
            train_loader = torch.load(os.path.join(preprocess_output_path,f'train_dataloader_{studyname}.pth'))
            test_loader = torch.load(os.path.join(preprocess_output_path,f'test_dataloader_{studyname}.pth'))
            if withval:
                val_loader = torch.load(os.path.join(preprocess_output_path,f'val_dataloader_{studyname}.pth'))
                
        trained_gae_model = build_gae_model(hyperparams=hyperparameters, state_dict= os.path.join(training_output_path,f'trained_model_{studyname}_{epochs}epochs.pth')).to(device)
        diffusion_model = build_diffusion_model(hyperparams=hyperparameters).to(device)

        wandb.init(project="DiffRNAFold", entity="diffrnafold",mode="online")

        diffusion_model.train_stable_diffusion(graphautoencoder=trained_gae_model, train_dataloader=train_loader, device=device)

        torch.save(diffusion_model.state_dict(), os.path.join(training_output_path, f'diffusion_model_{studyname}_{epochs}epochs.pth'))

        

if __name__ == "__main__":
    # wandb.agent("7vyprc9h", project="DiffFold-Sweep", entity="diffrnafold", function=main)
    main()