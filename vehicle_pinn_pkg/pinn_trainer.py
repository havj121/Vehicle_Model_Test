import torch
import torch.nn as nn
import numpy as np
import time

class PINN_Net(nn.Module):
    def __init__(self, output_dim, layers=[1, 64, 64]):
        super(PINN_Net, self).__init__()
        # Dynamic layers: Input(1) -> Hidden -> Output(output_dim)
        full_layers = layers + [output_dim]
        
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList()
        
        for i in range(len(full_layers)-1):
            self.linears.append(nn.Linear(full_layers[i], full_layers[i+1]))
        
        # Xavier Initialization
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, t):
        x = t
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x

class PINNTrainer:
    def __init__(self, vehicle, layers=[1, 64, 64], device='cpu'):
        self.vehicle = vehicle
        self.device = device
        
        # Automatically adapt output dimension based on vehicle model
        self.model = PINN_Net(output_dim=vehicle.state_dim, layers=layers).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_history = []
        
    def _compute_physics_loss(self, t_collocation, u):
        """
        Computes PDE residual loss.
        """
        t = t_collocation.clone().detach().requires_grad_(True)
        
        # Forward pass
        y_pred = self.model(t) # [batch, state_dim]
        
        # Automatic differentiation for all state variables
        y_t_list = []
        for i in range(self.vehicle.state_dim):
            y_i = y_pred[:, i:i+1]
            grad_y_i = torch.autograd.grad(y_i, t, grad_outputs=torch.ones_like(y_i), create_graph=True)[0]
            y_t_list.append(grad_y_i)
        
        y_t = torch.cat(y_t_list, dim=1) # [batch, state_dim]
        
        # Ensure u is properly formatted (e.g., list of tensors if multi-input)
        # We assume u is passed as a list of tensors or a single tensor, matching what compute_physics_residual expects
        residuals = self.vehicle.compute_physics_residual(t, y_pred, y_t, u)
        
        # Sum of squared residuals
        loss_pde = sum([torch.mean(res**2) for res in residuals])
        return loss_pde

    def _compute_data_loss(self, t_data, y_data):
        """
        Computes MSE loss against observed data.
        """
        if t_data is None or y_data is None:
            return torch.tensor(0.0).to(self.device)
            
        pred = self.model(t_data)
        loss_data = torch.mean((pred - y_data)**2)
        return loss_data

    def _compute_ic_loss(self, t0, y0):
        """
        Computes Initial Condition loss.
        """
        pred_0 = self.model(t0)
        loss_ic = torch.mean((pred_0 - y0)**2)
        return loss_ic

    def train(self, mode='hybrid', 
              epochs=2000, 
              t_train=None, y_train=None,  # Data
              t_collocation=None,          # Physics points
              u=None,                      # Control inputs (scalar, tensor, or list of scalars/tensors)
              ic_weight=1.0, pde_weight=1.0, data_weight=1.0,
              verbose=True):
        
        if verbose:
            print(f"Starting training in [{mode}] mode for {epochs} epochs...")
            print(f"Model: {self.vehicle.name}, State Dim: {self.vehicle.state_dim}")
            
        start_time = time.time()
        
        # Prepare tensors
        if t_train is not None and not isinstance(t_train, torch.Tensor):
            t_train = torch.tensor(t_train, dtype=torch.float32).reshape(-1, 1).to(self.device)
        if y_train is not None and not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, self.vehicle.state_dim).to(self.device)
        
        if t_collocation is not None and not isinstance(t_collocation, torch.Tensor):
            t_collocation = torch.tensor(t_collocation, dtype=torch.float32).reshape(-1, 1).to(self.device)
            
        # Initial condition (t=0)
        t0 = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
        # Try to infer y0 from training data if available, else assume zeros
        if y_train is not None:
            # Assuming first point is t=0
            y0_val = y_train[0:1, :]
        else:
            y0_val = torch.zeros((1, self.vehicle.state_dim), dtype=torch.float32).to(self.device)

        # Handle control inputs 'u'
        # If u is simple scalar/list, convert to tensor(s) for physics
        # For CombinedModel, u might be [Fx, delta]
        u_tensor = u
        if u is not None:
             if isinstance(u, (list, tuple)):
                 u_tensor = []
                 for val in u:
                     if not isinstance(val, torch.Tensor):
                         # If scalar, broadcast to collocation shape? Or just pass as scalar if residual function handles it
                         # Here we just pass it. The vehicle model's compute_physics_residual should handle scalars or tensors.
                         u_tensor.append(val)
                     else:
                         u_tensor.append(val)
             else:
                 # Single input
                 pass

        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            loss = torch.tensor(0.0).to(self.device)
            
            # 1. Data Loss
            if mode in ['data', 'hybrid'] and t_train is not None:
                loss_d = self._compute_data_loss(t_train, y_train)
                loss += data_weight * loss_d
            
            # 2. Physics Loss
            if mode in ['physics', 'hybrid'] and t_collocation is not None:
                loss_p = self._compute_physics_loss(t_collocation, u_tensor)
                loss += pde_weight * loss_p
                
            # 3. Initial Condition Loss
            if mode in ['physics', 'hybrid']:
                loss_i = self._compute_ic_loss(t0, y0_val)
                loss += ic_weight * loss_i
            
            loss.backward()
            self.optimizer.step()
            
            self.loss_history.append(loss.item())
            
            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
                
        if verbose:
            print(f"Training finished in {time.time()-start_time:.2f}s")

    def predict(self, t):
        self.model.eval()
        if not isinstance(t, torch.Tensor):
            t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1).to(self.device)
        else:
            t_tensor = t
        
        with torch.no_grad():
            pred = self.model(t_tensor).cpu().numpy()
        return pred
