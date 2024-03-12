import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np


class NeuralNet(nn.Module):

    def __init__(
        self,
        input_dimension,
        output_dimension,
        n_hidden_layers,
        neurons,
        regularization_param,
        regularization_exp,
        retrain_seed,
    ):
        super(NeuralNet, self).__init__()

        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = (
            nn.Tanh()
        )  # If you change this, remember to also change the init_weights function below
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)]
        )
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed
        # Random Seed for weight initialization
        self.init_xavier()

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain("tanh")
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if "weight" in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss


class BasePinn:
    def __init__(
        self,
        num_outputs,
        num_boundary_samples=64,
        num_interior_samples=512,
        device="cuda",
        seed=0,
        regularization_param=0.1
    ):
        self.num_boundary_samples = num_boundary_samples
        self.num_interior_samples = num_interior_samples

        self.device = device

        self.domain = torch.tensor([[0, 1], [0, 1]]).to(self.device)

        self.neural_network = NeuralNet(
            input_dimension=self.domain.shape[0],
            output_dimension=num_outputs,
            n_hidden_layers=5,
            neurons=256,
            regularization_param=regularization_param,
            regularization_exp=2.0,
            retrain_seed=seed,
        ).to(self.device)

        self.sobol_engine = torch.quasirandom.SobolEngine(
            dimension=self.domain.shape[0], seed=seed
        )

        # Weight of boundary terms
        self.lambda_u = 10

        self.cached_boundary = None
        self.cached_interior_points = self.sample_interior_points(use_cached=False)

        # Physical constants

        self.t_0 = 293.15  # Inflow Fluid Temperature

        self.p_0 = 1.0 # inflow pressure
        self.p_L = 0.0 # outflow pressure

    

        self.rho_l = 998  # Density Fluid

        self.C_f = 4180  # Heat capacity fluid
        self.v_f = 1.004e-3

        self.k_t = 0.598  # Thermal conductivity Fluid
        self.k_b = 149  # Thermal conductivity Solid

        self.H_t = 0.5 * 380e-6  # Half-Thickness Fluid
        self.H_b = 0.5 * 150e-6  # Half-Thickness Solid

        self.h_b = self.k_b / self.H_b
        self.h_t = 35 * self.k_t / (26 * self.H_t)

        self.q_in = 5e5  # heat-source die layer

        applied_pressure = 100000 * 5.0 / 4.0  # kg m^-1 s^-2
        self.L = 0.001  # m
        self.U = np.sqrt(applied_pressure / self.rho_l)  # m/s

        self.nu = self.v_f / self.rho_l

        self.Re = self.L * self.U / self.nu
        
        self.q_k = 1 #this change in each optimization steps
        self.alpha_f = 0
        self.alpha_s = 5*self.L**2/(2*self.H_t**2)

    def eval(self, points):

        # Temp PINN
        if self.num_outputs == 2:
            mean = torch.tensor([293.2327663042587, 293.1548959834691], device=self.device)
            std = torch.tensor(
                [0.05894596474036041, 0.0043651020561740125], device=self.device
            )

        # Flow PINN
        if self.num_outputs == 3:
            mean = torch.tensor([0.5535466316835341, 6.359693982990098, 6.359693982990098], device=self.device)
            std = torch.tensor(
                [0.2972179586141333, 4.127014962188644, 4.127014962188644], device=self.device
            )

        return self.neural_network(points) * std + mean

    # Converts point in [0, 1] reference domain to true domain
    def _to_domain(self, points):
        # points dim = (batch_size, dim(domain))
        assert points.shape[1] == self.domain.shape[0]
        return points * (self.domain[:, 1] - self.domain[:, 0]) + self.domain[:, 0]

    # Sample points in domain
    def sample_domain(self, n):
        return self._to_domain(self.sobol_engine.draw(n).to(self.device))

    # Sample inflow
    def sample_inflow_points(self, n_samples):
        samples = self.sample_domain(n_samples)
        samples[:, 0] = torch.full(samples[:, 0].shape, self.domain[0, 0])

        return samples

    # Sample outflow
    def sample_outflow_points(self, n_samples):
        samples = self.sample_domain(n_samples)
        samples[:, 0] = torch.full(samples[:, 0].shape, self.domain[0, 1])

        return samples

    def sample_wall_points(self, n_samples):
        samples = self.sample_domain(n_samples)

        # split half of the sample to top and half to bottom walls
        half_samples = n_samples // 2
        middle = n_samples - half_samples

        samples[:middle, 1] = torch.full(samples[:middle, 1].shape, self.domain[1, 0])
        samples[middle:, 1] = torch.full(samples[middle:, 1].shape, self.domain[1, 1])

        return samples

    def sample_interior_points(self, use_cached=False):
        if use_cached:
            return self.cached_interior_points

        return self.sample_domain(self.num_interior_samples)

    def compute_boundary_error(self, use_cached=False):
        raise NotImplementedError(
            "Implement the boundary error function for the base class"
        )

    # Compute error with some input/output pair e.g. on the boundaries or as part of supervised training
    def compute_supervised_error(self, points, values):
        raise NotImplementedError(
            "Implement the supervised error function for the base class"
        )

    # Compute error according to the PDE of the interior
    def compute_unsupervised_error(self, points, power_map=None, flow_field=None, rho=None):
        raise NotImplementedError(
            "Implement the unsupervised error function for the base class"
        )

    def fit(
        self,
        num_epochs,
        optimizer,
        verbose=False,
        data=None,  # torch dataloader
        use_cached_data=True,
        power_map=None,
        flow_field=None,
        rho=None
    ):
        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose:
                print(
                    "################################ ",
                    epoch,
                    " ################################",
                )

            if data is not None:
                for i, (data_input, data_output) in enumerate(data):

                    def closure():
                        optimizer.zero_grad()
                        diff_loss = 10e5 * torch.mean(
                            self.compute_supervised_error(
                                points=data_input, values=data_output
                            )
                            ** 2
                        )

                        reg_loss = self.neural_network.regularization()

                        loss = diff_loss + reg_loss

                        loss.backward()
                        history.append(loss.item())

                        if verbose:
                            print(
                                "Total loss: ",
                                round(loss.item(), 4),
                                "| Difference Error: ",
                                round(diff_loss.item(), 4),
                                "| Regression Error: ",
                                round(reg_loss.item(), 4),
                            )

                        return loss

                    optimizer.step(closure=closure)
            else:

                interior_training_data = DataLoader(
                    TensorDataset(self.sample_interior_points(use_cached_data)),
                    batch_size=self.num_interior_samples,
                    shuffle=False,
                )

                for interior_points, in interior_training_data:
                    def closure():
                        optimizer.zero_grad()
                        loss_boundary = torch.mean(
                            self.compute_boundary_error() ** 2
                        )

                        loss_interior = torch.mean(  # 10e-10
                            self.compute_unsupervised_error(
                                points=interior_points,
                                power_map=power_map,
                                flow_field=flow_field,
                                rho=rho
                            )
                            ** 2
                        )

                        loss_reg = self.neural_network.regularization()

                        loss = torch.log(
                            self.lambda_u * loss_boundary + loss_interior + loss_reg
                        )

                        loss.backward()
                        history.append(loss.item())
                        if verbose:
                            print(
                                "Total loss: ",
                                round(loss.item(), 4),
                                "| Boundary Error: ",
                                round(self.lambda_u * loss_boundary.item(), 4),
                                "| Interior Error: ",
                                round(loss_interior.item(), 4),
                                "| Regularization Error: ",
                                round(loss_reg.item(), 4),
                            )
                        return loss

                    optimizer.step(closure=closure)

        print("Final Loss: ", history[-1])

        return history
