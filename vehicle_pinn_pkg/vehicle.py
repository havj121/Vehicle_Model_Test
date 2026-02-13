import numpy as np
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod

class BaseVehicleModel(ABC):
    """
    Abstract base class for vehicle models.
    """
    def __init__(self, name="BaseVehicle"):
        self.name = name
        # Default state and input dimensions (to be overridden)
        self.state_dim = 0
        self.input_dim = 0
        self.state_names = []
        self.input_names = []

    @abstractmethod
    def get_dynamics(self, t, y, u):
        """
        Compute dy/dt = f(t, y, u)
        """
        pass

    @abstractmethod
    def compute_physics_residual(self, t, y, y_t, u):
        """
        Compute physics residual for PINN loss: y_t - f(t, y, u)
        Returns a list of residuals for each state dimension.
        """
        pass

    def generate_ground_truth(self, t_span, y0, u, n_points=1000):
        """
        Generates ground truth data using numerical integration.
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(
            lambda t, y: self.get_dynamics(t, y, u),
            t_span, y0, t_eval=t_eval, method='RK45'
        )
        return sol.t, sol.y.T  # t: [N], y: [N, state_dim]


class LateralModel(BaseVehicleModel):
    """
    Linear Bicycle Model (2 DOF): Sideslip (beta) and Yaw Rate (r)
    State: [beta, r]
    Input: [delta] (Steering angle)
    """
    def __init__(self, m=1500.0, Iz=3000.0, a=1.2, b=1.5, Cf=80000.0, Cr=80000.0, Vx=20.0):
        super().__init__("LateralModel")
        self.m = m
        self.Iz = Iz
        self.a = a
        self.b = b
        self.Cf = Cf
        self.Cr = Cr
        self.Vx = Vx
        
        self.state_dim = 2
        self.input_dim = 1
        self.state_names = ['Beta (rad)', 'Yaw Rate (rad/s)']
        self.input_names = ['Steering Angle (rad)']
        
        # Coefficients
        self.a11 = -(Cf + Cr) / (m * Vx)
        self.a12 = -1.0 + (-a * Cf + b * Cr) / (m * Vx**2)
        self.a21 = -(a * Cf - b * Cr) / Iz
        self.a22 = -(a**2 * Cf + b**2 * Cr) / (Iz * Vx)
        self.b1 = Cf / (m * Vx)
        self.b2 = a * Cf / Iz

    def get_dynamics(self, t, y, u):
        beta, r = y
        delta = u[0] if isinstance(u, (list, tuple, np.ndarray)) else u
        
        dbeta_dt = self.a11 * beta + self.a12 * r + self.b1 * delta
        dr_dt = self.a21 * beta + self.a22 * r + self.b2 * delta
        return [dbeta_dt, dr_dt]

    def compute_physics_residual(self, t, y, y_t, u):
        # y: [batch, 2], y_t: [batch, 2], u: [batch, 1] or scalar
        beta = y[:, 0:1]
        r = y[:, 1:2]
        beta_t = y_t[:, 0:1]
        r_t = y_t[:, 1:2]
        
        delta = u # u is expected to be a tensor or scalar compatible with broadcasting
        
        res_beta = beta_t - (self.a11 * beta + self.a12 * r + self.b1 * delta)
        res_r = r_t - (self.a21 * beta + self.a22 * r + self.b2 * delta)
        
        return [res_beta, res_r]


class LongitudinalModel(BaseVehicleModel):
    """
    Simple Longitudinal Model (1 DOF): Velocity (v)
    State: [v]
    Input: [F_x] (Longitudinal Force)
    Equation: m * dv/dt = F_x - F_aero - F_roll
    """
    def __init__(self, m=1500.0, Cd=0.3, A=2.2, rho=1.225, f=0.015):
        super().__init__("LongitudinalModel")
        self.m = m
        self.Cd = Cd
        self.A = A
        self.rho = rho
        self.f = f
        self.g = 9.81
        
        self.state_dim = 1
        self.input_dim = 1
        self.state_names = ['Velocity (m/s)']
        self.input_names = ['Force (N)']

    def get_dynamics(self, t, y, u):
        v = y[0]
        Fx = u[0] if isinstance(u, (list, tuple, np.ndarray)) else u
        
        F_aero = 0.5 * self.rho * self.Cd * self.A * v**2
        F_roll = self.m * self.g * self.f
        
        dv_dt = (Fx - F_aero - F_roll) / self.m
        return [dv_dt]

    def compute_physics_residual(self, t, y, y_t, u):
        v = y[:, 0:1]
        v_t = y_t[:, 0:1]
        Fx = u
        
        F_aero = 0.5 * self.rho * self.Cd * self.A * v**2
        F_roll = self.m * self.g * self.f
        
        res_v = v_t - (Fx - F_aero - F_roll) / self.m
        return [res_v]


class CombinedModel(BaseVehicleModel):
    """
    Combined Lateral & Longitudinal Model (3 DOF)
    State: [x, y, psi, v, beta, r] -> Simplified to [v, beta, r] for dynamics
    Let's keep it simple: [v, beta, r]
    Input: [Fx, delta]
    """
    def __init__(self, m=1500.0, Iz=3000.0, a=1.2, b=1.5, Cf=80000.0, Cr=80000.0, 
                 Cd=0.3, A=2.2, rho=1.225, f=0.015):
        super().__init__("CombinedModel")
        # Reuse parameters
        self.lat_model = LateralModel(m, Iz, a, b, Cf, Cr, Vx=1.0) # Vx will be dynamic
        self.long_model = LongitudinalModel(m, Cd, A, rho, f)
        
        self.state_dim = 3
        self.input_dim = 2
        self.state_names = ['Velocity (m/s)', 'Beta (rad)', 'Yaw Rate (rad/s)']
        self.input_names = ['Force (N)', 'Steering Angle (rad)']
        
        self.m = m
        self.Iz = Iz
        self.a = a
        self.b = b
        self.Cf = Cf
        self.Cr = Cr

    def get_dynamics(self, t, y, u):
        v, beta, r = y
        Fx, delta = u
        
        # Longitudinal Dynamics
        dv_dt = self.long_model.get_dynamics(t, [v], [Fx])[0]
        
        # Lateral Dynamics (Coupled with v)
        # Note: Avoid division by zero for v
        v_safe = v if abs(v) > 0.1 else 0.1
        
        # Recalculate coefficients based on current v
        a11 = -(self.Cf + self.Cr) / (self.m * v_safe)
        a12 = -1.0 + (-self.a * self.Cf + self.b * self.Cr) / (self.m * v_safe**2)
        a21 = -(self.a * self.Cf - self.b * self.Cr) / self.Iz
        a22 = -(self.a**2 * self.Cf + self.b**2 * self.Cr) / (self.Iz * v_safe)
        b1 = self.Cf / (self.m * v_safe)
        b2 = self.a * self.Cf / self.Iz
        
        dbeta_dt = a11 * beta + a12 * r + b1 * delta
        dr_dt = a21 * beta + a22 * r + b2 * delta
        
        return [dv_dt, dbeta_dt, dr_dt]

    def compute_physics_residual(self, t, y, y_t, u):
        v = y[:, 0:1]
        beta = y[:, 1:2]
        r = y[:, 2:3]
        
        v_t = y_t[:, 0:1]
        beta_t = y_t[:, 1:2]
        r_t = y_t[:, 2:3]
        
        Fx = u[0] # Assumes u is list of tensors [Fx, delta]
        delta = u[1]
        
        # Longitudinal Residual
        F_aero = 0.5 * self.long_model.rho * self.long_model.Cd * self.long_model.A * v**2
        F_roll = self.long_model.m * self.long_model.g * self.long_model.f
        res_v = v_t - (Fx - F_aero - F_roll) / self.long_model.m
        
        # Lateral Residuals
        # Need to handle v in denominator carefully for tensors
        # Add epsilon for numerical stability
        v_safe = torch.where(torch.abs(v) > 0.1, v, torch.tensor(0.1).to(v.device))
        
        a11 = -(self.Cf + self.Cr) / (self.m * v_safe)
        a12 = -1.0 + (-self.a * self.Cf + self.b * self.Cr) / (self.m * v_safe**2)
        a21 = -(self.a * self.Cf - self.b * self.Cr) / self.Iz
        a22 = -(self.a**2 * self.Cf + self.b**2 * self.Cr) / (self.Iz * v_safe)
        b1 = self.Cf / (self.m * v_safe)
        b2 = self.a * self.Cf / self.Iz
        
        res_beta = beta_t - (a11 * beta + a12 * r + b1 * delta)
        res_r = r_t - (a21 * beta + a22 * r + b2 * delta)
        
        return [res_v, res_beta, res_r]
