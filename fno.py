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

import simulation_hackaton_eth.transform as transform
import simulation_hackaton_eth.dataprocessor as dataprocessor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32,
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
        positional_encoding=True
)

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

        u_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        u_encoder.fit(u)
    else:
        output_encoder = None
    
    if positional_encoding:
        pos_encoding = transform.PositionalEmbedding2D(grid_boundaries)

    data_processor = dataprocessor.DefaultDataProcessor(in_normalizer=input_encoder,
                                          out_normalizer=output_encoder,
                                          positional_encoding=pos_encoding)
    
    return data_processor

def import_data(file_path:str, 
                a_file:str, 
                sol_file :str, 
                train_test_split : bool = True, 
                gaussian_normalization: bool = True,
                grid_boundaries: np.array = [[0, 1], [0, 1]],
                batch_size: int = 8,
                num_workers : int = 8,
                pin_memory : bool = False,
                persistent_workers :bool = False):
    """Function to import the data assuming the file are
    npy file with the data.shape = (nbr_data_points, nbr_observations, resolution_x_coord, resolution_y_coord) """
    a = np.load(file_path + a_file)
    u = np.load(file_path + sol_file)
    
    if train_test_split:
        
        a_train, u_train, a_test, u_test = train_test_split(a, u, test_size = 0.2, random_state = 42)
        a_train = torch.Tensor(a_train).to(device)
        u_train = torch.Tensor(u_train).to(device)
        a_test = torch.Tensor(a_test).to(device)
        u_test = torch.Tensor(u_test).to(device)
        
        if gaussian_normalization:
           data_processor = encode_data(a_train, u_train, grid_boundaries=grid_boundaries)
        else:
            data_processor = None
        
        train_db, test_db = torch.utils.data.TensorDataset(a_train, u_train), torch.utils.data.TensorDataset(a_test, u_test)
        train_loader = torch.utils.data.DataLoader(train_db,
                                               batch_size=batch_size, shuffle=True, drop_last=True,
                                               num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        test_loader = torch.utils.data.DataLoader(test_db,
                                               batch_size=batch_size, shuffle=True, drop_last=True,
                                               num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        
        return train_loader, test_loader , data_processor

    a = Tensor(a).to(device)
    u = Tensor(u).to(device)
    
    if gaussian_normalization:
        data_processor = encode_data(a,u)
    else:
        data_processor = None
    
    db = torch.utils.data.TensorDataset(a, u)
    loader = torch.utils.data.DataLoader(db,
                                        batch_size=batch_size, shuffle=True, drop_last=True,
                                        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        
    
    return loader, data_processor

file_path = ""
a_file = ""
sol_file = ""

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

try:
    train_loader, test_loader, data_processor = import_data(file_path=file_path,
                                                        a_file=a_file,
                                                        sol_file=sol_file,
                                                        train_test_split = True, #change this to false if dont want train_test split
                                                        gaussian_normalization = False,
                                                        grid_boundaries= [[0, 1], [0, 1]],
                                                        batch_size=8,
                                                        num_workers= 8,
                                                        pin_memory=pin_memory,
                                                        persistent_workers=False,  
    )
except:
    loader, data_processor = import_data(file_path=file_path,
                                                        a_file=a_file,
                                                        sol_file=sol_file,
                                                        train_test_split = False,
                                                        gaussian_normalization = False,
                                                        grid_boundaries= [[0, 1], [0, 1]],
                                                        batch_size=8,
                                                        num_workers= 8,
                                                        pin_memory=pin_memory,
                                                        persistent_workers=False,  
    )

print(data_processor)
data_processor = data_processor.to(device)

model = TFNO(n_modes=(16, 16),
             hidden_channels=32,
             in_channels = 3,
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
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)

test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0:
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.squeeze())
    if index == 0:
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.savefig('fno_out.png')