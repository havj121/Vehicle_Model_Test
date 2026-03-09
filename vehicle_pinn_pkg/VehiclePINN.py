import numpy as np
import torch
import matplotlib.pyplot as plt
from .vehicle import BicycleModel, LongitudinalModel, CombinedModel
from .pinn_trainer import PINNTrainer

class VehiclePINN:
    """
    Main class that orchestrates the Vehicle Model and PINN Trainer.
    """
    def __init__(self, model_type='lateral', device='cpu', **vehicle_params):
        """
        model_type: 'lateral', 'longitudinal', or 'combined'
        """
        self.device = device
        self.model_type = model_type.lower()
        
        if self.model_type == 'lateral':
            self.vehicle = BicycleModel(**vehicle_params)
        elif self.model_type == 'longitudinal':
            self.vehicle = LongitudinalModel(**vehicle_params)
        elif self.model_type == 'combined':
            self.vehicle = CombinedModel(**vehicle_params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'lateral', 'longitudinal', or 'combined'.")
            
        self.trainer = PINNTrainer(self.vehicle, device=device)
        
    def generate_data(self, t_span, y0, u, n_points=1000):
        """
        Generate ground truth data using the vehicle model.
        """
        return self.vehicle.generate_ground_truth(t_span, y0, u, n_points)
    
    def train(self, mode='hybrid', **kwargs):
        """
        Train the PINN model.
        """
        self.trainer.train(mode=mode, **kwargs)
        
    def predict(self, t):
        """
        Predict using the trained PINN model.
        """
        return self.trainer.predict(t)
    
    def visualize(self, t_true, y_true, t_pred, y_pred, title_suffix=""):
        """
        Visualize the comparison results dynamically based on state dimension.
        """
        state_dim = self.vehicle.state_dim
        state_names = self.vehicle.state_names
        
        plt.figure(figsize=(6 * state_dim, 5))
        
        for i in range(state_dim):
            plt.subplot(1, state_dim, i+1)
            plt.plot(t_true, y_true[:, i], 'k-', label='Ground Truth')
            plt.plot(t_true, y_pred[:, i], 'r--', label='Prediction')
            plt.xlabel('Time (s)')
            plt.ylabel(state_names[i])
            plt.title(f'{state_names[i]} {title_suffix}')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
