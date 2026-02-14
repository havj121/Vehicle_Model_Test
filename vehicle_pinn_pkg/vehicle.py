import numpy as np
import torch
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod

class BaseVehicleModel(ABC):
    """
    Abstract base class for vehicle models;
    subclasses must implement get_dynamics() and get_dynamics_torch().
    """
    def __init__(self, name="BaseVehicle"):
        self.name = name
        # Default state and input dimensions (to be overridden)
        self.state_dim = 0
        self.input_dim = 0
        self.state_names = []
        self.input_names = []
        self.parameters = {}  # {key: {'value': val, 'name': name, 'unit': unit}}

    def set_parameters(self, new_params):
        """
        Update vehicle parameters and trigger coefficient update.
        new_params: dict, e.g., {'m': 1600.0, 'Vx': 25.0}
        """
        for key, value in new_params.items():
            if key in self.parameters:
                self.parameters[key]['value'] = value
            else:
                print(f"Warning: Parameter '{key}' not found in {self.name}")
        
        # Automatically update internal coefficients or dependent states
        self._update_coefficients()

    def get_parameters(self):
        """
        Returns all parameters of the model.
        Returns: dict where key is variable name, and value contains name, value, and unit.
        """
        return self.parameters

    def _check_input_dims(self, x, u):
        """
        Check if dimensions of state x and input u match model definitions.
        x: state vector (numpy array or torch tensor)
        u: input vector (scalar, list, numpy array or torch tensor)
        """
        # 1. Check State x
        if isinstance(x, (np.ndarray, list)):
            x_len = len(x)
        elif torch.is_tensor(x):
            # For torch, check the last dimension if it's a batch [N, state_dim]
            x_len = x.shape[-1]
        else:
            x_len = 1 if np.isscalar(x) else 0

        if x_len != self.state_dim:
            raise ValueError(
                f"[{self.name}] State dimension mismatch!\n"
                f"Expected: {self.state_dim} (Names: {self.state_names})\n"
                f"Received dimension: {x_len}"
            )

        # 2. Check Input u
        if isinstance(u, (list, np.ndarray)):
            u_len = len(u)
        elif torch.is_tensor(u):
            # For torch, handle both scalar [N, 1] and vector [N, input_dim]
            u_len = u.shape[-1] if u.dim() > 0 else 1
        else:
            u_len = 1 if np.isscalar(u) else 0

        if u_len != self.input_dim:
            raise ValueError(
                f"[{self.name}] Input dimension mismatch!\n"
                f"Expected: {self.input_dim} (Names: {self.input_names})\n"
                f"Received dimension: {u_len}"
            )

    def _update_coefficients(self):
        """
        Optional: Update pre-calculated coefficients if the model uses them.
        To be overridden by subclasses if needed.
        """
        pass

    @abstractmethod
    def get_dynamics(self, t, x, u):
        """
        Compute dx/dt = f(t, x, u) for numerical integration (numpy/scalar).
        """
        pass

    @abstractmethod
    def get_dynamics_torch(self, t, x, u):
        """
        Compute dx/dt = f(t, x, u) for PINN physics loss (torch tensors).
        Returns a list of tensors [dx1_dt, dx2_dt, ...] corresponding to state_dim.
        """
        pass

    def generate_ground_truth(self, t_span, x0, u, n_points=1000):
        """
        Generates ground truth data using numerical integration.
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(
            lambda t, x: self.get_dynamics(t, x, u),
            t_span, x0, t_eval=t_eval, method='RK45'
        )
        return sol.t, sol.y.T  # t: [N], y: [N, state_dim]


class BicycleModel(BaseVehicleModel):
    """
    Linear Bicycle Model (2 DOF): Sideslip (beta) and Yaw Rate (r)
    State: [beta, r]
    Input: [delta] (Steering angle)
    """
    def __init__(self, m=1500.0, Iz=3000.0, a=1.2, b=1.5, Cf=80000.0, Cr=80000.0, Vx=20.0):
        super().__init__("BicycleModel")
        self.parameters = {
            'm': {'value': m, 'name': 'mass', 'unit': 'kg'},
            'Iz': {'value': Iz, 'name': 'yaw_moment_of_inertia', 'unit': 'kg*m^2'},
            'a': {'value': a, 'name': 'distance_cg_front', 'unit': 'm'},
            'b': {'value': b, 'name': 'distance_cg_rear', 'unit': 'm'},
            'Cf': {'value': Cf, 'name': 'cornering_stiffness_front', 'unit': 'N/rad'},
            'Cr': {'value': Cr, 'name': 'cornering_stiffness_rear', 'unit': 'N/rad'},
            'Vx': {'value': Vx, 'name': 'longitudinal_velocity', 'unit': 'm/s'}
        }
        
        self.state_dim = 2
        self.input_dim = 1
        self.state_names = ['Beta (rad)', 'Yaw Rate (rad/s)']
        self.input_names = ['Steering Angle (rad)']
        
        # Coefficients (calculated once for efficiency in linear model)
        self._update_coefficients()

    def _update_coefficients(self):
        p = {k: v['value'] for k, v in self.parameters.items()}
        m, Iz, a, b, Cf, Cr, Vx = p['m'], p['Iz'], p['a'], p['b'], p['Cf'], p['Cr'], p['Vx']
        
        self.a11 = -(Cf + Cr) / (m * Vx)
        self.a12 = -1.0 + (-a * Cf + b * Cr) / (m * Vx**2)
        self.a21 = -(a * Cf - b * Cr) / Iz
        self.a22 = -(a**2 * Cf + b**2 * Cr) / (Iz * Vx)
        self.b1 = Cf / (m * Vx)
        self.b2 = a * Cf / Iz

    def get_dynamics(self, t, x, u):
        self._check_input_dims(x, u)
        beta, r = x
        # steering angle
        delta = u
        
        dbeta_dt = self.a11 * beta + self.a12 * r + self.b1 * delta
        dr_dt = self.a21 * beta + self.a22 * r + self.b2 * delta
        return [dbeta_dt, dr_dt]

    def get_dynamics_torch(self, t, x, u):
        # We skip check_input_dims in torch version for performance 
        # as it's called every iteration, or we could keep it if desired.
        # x: [batch, 2], u: tensor or scalar
        beta = x[:, 0:1]
        r = x[:, 1:2]
        delta = u
        
        dbeta_dt = self.a11 * beta + self.a12 * r + self.b1 * delta
        dr_dt = self.a21 * beta + self.a22 * r + self.b2 * delta
        return [dbeta_dt, dr_dt]


class LongitudinalModel(BaseVehicleModel):
    """
    Simple Longitudinal Model (1 DOF): Velocity (v)
    State: [v]
    Input: [F_x] (Longitudinal Force)
    Equation: m * dv/dt = F_x - F_aero - F_roll
    """
    def __init__(self, m=1500.0, Cd=0.3, A=2.2, rho=1.225, f=0.015):
        super().__init__("LongitudinalModel")
        self.parameters = {
            'm': {'value': m, 'name': 'mass', 'unit': 'kg'},
            'Cd': {'value': Cd, 'name': 'drag_coefficient', 'unit': '-'},
            'A': {'value': A, 'name': 'frontal_area', 'unit': 'm^2'},
            'rho': {'value': rho, 'name': 'air_density', 'unit': 'kg/m^3'},
            'f': {'value': f, 'name': 'rolling_resistance_coefficient', 'unit': '-'},
            'g': {'value': 9.81, 'name': 'gravity', 'unit': 'm/s^2'}
        }
        
        self.state_dim = 1
        self.input_dim = 1
        self.state_names = ['Velocity (m/s)']
        self.input_names = ['Force (N)']
        self._update_coefficients()

    def _update_coefficients(self):
        # Longitudinal model doesn't have pre-calculated linear coefficients
        # but we could add logic here if needed (e.g., updating lookup tables)
        pass

    def get_dynamics(self, t, x, u):
        self._check_input_dims(x, u)
        v = x[0]
        Fx = u[0] if isinstance(u, (list, tuple, np.ndarray)) else u
        
        p = {k: v['value'] for k, v in self.parameters.items()}
        F_aero = 0.5 * p['rho'] * p['Cd'] * p['A'] * v**2
        F_roll = p['m'] * p['g'] * p['f']
        
        dv_dt = (Fx - F_aero - F_roll) / p['m']
        return [dv_dt]

    def get_dynamics_torch(self, t, x, u):
        p = {k: v['value'] for k, v in self.parameters.items()}
        # x: [batch, 1], u: tensor/scalar
        v = x[:, 0:1]
        Fx = u
        
        F_aero = 0.5 * p['rho'] * p['Cd'] * p['A'] * v**2
        F_roll = p['m'] * p['g'] * p['f']
        
        dv_dt = (Fx - F_aero - F_roll) / p['m']
        return [dv_dt]


class CombinedModel(BaseVehicleModel):
    """
    Combined Lateral & Longitudinal Model (3 DOF)
    State: [v, beta, r]
    Input: [Fx, delta]
    """
    def __init__(self, m=1500.0, Iz=3000.0, a=1.2, b=1.5, Cf=80000.0, Cr=80000.0, 
                 Cd=0.3, A=2.2, rho=1.225, f=0.015):
        super().__init__("CombinedModel")
        self.parameters = {
            'm': {'value': m, 'name': 'mass', 'unit': 'kg'},
            'Iz': {'value': Iz, 'name': 'yaw_moment_of_inertia', 'unit': 'kg*m^2'},
            'a': {'value': a, 'name': 'distance_cg_front', 'unit': 'm'},
            'b': {'value': b, 'name': 'distance_cg_rear', 'unit': 'm'},
            'Cf': {'value': Cf, 'name': 'cornering_stiffness_front', 'unit': 'N/rad'},
            'Cr': {'value': Cr, 'name': 'cornering_stiffness_rear', 'unit': 'N/rad'},
            'Cd': {'value': Cd, 'name': 'drag_coefficient', 'unit': '-'},
            'A': {'value': A, 'name': 'frontal_area', 'unit': 'm^2'},
            'rho': {'value': rho, 'name': 'air_density', 'unit': 'kg/m^3'},
            'f': {'value': f, 'name': 'rolling_resistance_coefficient', 'unit': '-'},
            'g': {'value': 9.81, 'name': 'gravity', 'unit': 'm/s^2'}
        }
        
        self.state_dim = 3
        self.input_dim = 2
        self.state_names = ['Velocity (m/s)', 'Beta (rad)', 'Yaw Rate (rad/s)']
        self.input_names = ['Force (N)', 'Steering Angle (rad)']
        self._update_coefficients()

    def _update_coefficients(self):
        # Combined model uses parameters directly in dynamics, 
        # but we call this to maintain consistency and allow future extensions.
        pass

    def get_dynamics(self, t, x, u):
        self._check_input_dims(x, u)
        p = {k: v['value'] for k, v in self.parameters.items()}
        v, beta, r = x
        Fx, delta = u
        
        # Longitudinal
        F_aero = 0.5 * p['rho'] * p['Cd'] * p['A'] * v**2
        F_roll = p['m'] * p['g'] * p['f']
        dv_dt = (Fx - F_aero - F_roll) / p['m']
        
        # Lateral (Coupled)
        v_safe = v if abs(v) > 0.1 else 0.1
        a11 = -(p['Cf'] + p['Cr']) / (p['m'] * v_safe)
        a12 = -1.0 + (-p['a'] * p['Cf'] + p['b'] * p['Cr']) / (p['m'] * v_safe**2)
        a21 = -(p['a'] * p['Cf'] - p['b'] * p['Cr']) / p['Iz']
        a22 = -(p['a']**2 * p['Cf'] + p['b']**2 * p['Cr']) / (p['Iz'] * v_safe)
        b1 = p['Cf'] / (p['m'] * v_safe)
        b2 = p['a'] * p['Cf'] / p['Iz']
        
        dbeta_dt = a11 * beta + a12 * r + b1 * delta
        dr_dt = a21 * beta + a22 * r + b2 * delta
        
        return [dv_dt, dbeta_dt, dr_dt]

    def get_dynamics_torch(self, t, x, u):
        p = {k: v['value'] for k, v in self.parameters.items()}
        # x: [batch, 3], u: list of tensors [Fx, delta]
        v = x[:, 0:1]
        beta = x[:, 1:2]
        r = x[:, 2:3]
        
        Fx = u[0]
        delta = u[1]
        
        # Longitudinal
        F_aero = 0.5 * p['rho'] * p['Cd'] * p['A'] * v**2
        F_roll = p['m'] * p['g'] * p['f']
        dv_dt = (Fx - F_aero - F_roll) / p['m']
        
        # Lateral
        v_safe = torch.where(torch.abs(v) > 0.1, v, torch.tensor(0.1).to(v.device))
        
        a11 = -(p['Cf'] + p['Cr']) / (p['m'] * v_safe)
        a12 = -1.0 + (-p['a'] * p['Cf'] + p['b'] * p['Cr']) / (p['m'] * v_safe**2)
        a21 = -(p['a'] * p['Cf'] - p['b'] * p['Cr']) / p['Iz']
        a22 = -(p['a']**2 * p['Cf'] + p['b']**2 * p['Cr']) / (p['Iz'] * v_safe)
        b1 = p['Cf'] / (p['m'] * v_safe)
        b2 = p['a'] * p['Cf'] / p['Iz']
        
        dbeta_dt = a11 * beta + a12 * r + b1 * delta
        dr_dt = a21 * beta + a22 * r + b2 * delta
        
        return [dv_dt, dbeta_dt, dr_dt]
