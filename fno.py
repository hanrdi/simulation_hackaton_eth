import torch
import matplotlib.pyplot as plt
import sys
import numpy as np

from sklearn.model_selection import train_test_split

from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

import transform 
import dataprocessor
from load_data import ProcessData

device = torch.device('cpu')
print(device)

#train_loader, test_loaders, data_processor = load_darcy_flow_small(
#        n_train=1000, batch_size=32,
#        test_resolutions=[16, 32], n_tests=[100, 50],
#        test_batch_sizes=[32, 32],
#        positional_encoding=True
#)

def encode_data(a: torch.Tensor,
                u : torch.Tensor ,
                encoding: str = 'pixel-wise', 
                encode_a: bool = True, 
                encode_u: bool = True,
                grid_boundaries: np.array = [[0, 1], [0, 1]],
                ):
    """Function to encode the input or output as N(0,1) distributed data"""
    pos_encoding = None
    if encode_a:
        if encoding == 'channel-wise':
            reduce_dims = list(range(x_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        a_encoder = transform.UnitGaussianNormalizer(dim=reduce_dims)
        a_encoder.fit(a)
    else:
        a_encoder = None

    if encode_u:
        if encoding == 'channel-wise':
            reduce_dims = list(range(y_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        u_encoder = transform.UnitGaussianNormalizer(dim=reduce_dims)
        u_encoder.fit(u)
    else:
        output_encoder = None
    
    if positional_encoding:
        pos_encoding = transform.PositionalEmbedding2D(grid_boundaries)

    data_processor = dataprocessor.DefaultDataProcessor(in_normalizer=input_encoder,
                                          out_normalizer=output_encoder,
                                          positional_encoding=pos_encoding)
    return data_processor


def load_data():
    file_path = "/mnt/c/Users/bonvi/Documents/simulation_hack/simulation_hackaton_eth-rafael/simulation_hackaton_eth-rafael/"
    num_of_files = 2
    process_data = ProcessData(file_path, num_of_files)
    dic, dic_power_map = process_data.load_data(file_path)
    dic_data = process_data.import_data(dic_file = dic, per_file = False)
    dic_power_map_data = process_data.import_data(dic_file = dic_power_map, per_file = True)
    #print(dic_power_map_data.keys())
        
        #process_data.plot_data(dic_data)
    tensor_data = process_data.data_to_tensor(dic_data, dic_power_map_data)
    
    return tensor_data


if device == 'cpu':
    pin_memory = False
else:
    pin_memory = True
    




# Have to change 
# file_path
# a_file
# sol_file
# gaussian_normalization
# grid_boundaries

#data_processor = data_processor.to(device)

loader = load_data()

model = TFNO(n_modes=(100,100),
             hidden_channels=32,
             in_channels = 1,
             out_channels = 1,
             n_layers = 4,
             projection_channels=64,
             factorization='tucker',
             rank=0.42)

model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()

optimizer = torch.optim.Adam(model.parameters(),
                                lr=8e-3,
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}

print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()

trainer = Trainer(model=model, 
                  n_epochs=20,
                  device=device,
                  #data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)

trainer.train(train_loader=loader,
              test_loaders=loader,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)
