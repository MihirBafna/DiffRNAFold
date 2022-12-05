import wandb
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'val_loss'
		},
    'parameters': {
        'lr': {'values': [0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001]},
        'latent_dim':{'values':[64, 128, 256, 512, 1024]},
        'model_type':{'values': ['VAE', 'AE']},
        'augmented': {'values': [True, False]},
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Difffold-Sweep", entity="diffrnafold")