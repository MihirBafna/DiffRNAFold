import wandb
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'val_loss'
		},
    'parameters': {
        'lr': {'max': 0.1, 'min': 0.0000001},
        'latent_dim':{'values':[64, 128, 256, 512, 1024]},
        'model_type':{'values': ['VAE', 'AE']}
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Difffold-Sweep", entity="diffrnafold")