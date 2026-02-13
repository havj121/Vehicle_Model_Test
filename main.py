import numpy as np
import torch
import matplotlib.pyplot as plt
from vehicle_pinn_pkg import VehiclePINN

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # =========================================================
    # Scenario 1: Lateral Model (Original)
    # =========================================================
    print("\n================================================")
    print("Scenario 1: Lateral Dynamics (Beta, Yaw Rate)")
    print("================================================")
    
    vp_lat = VehiclePINN(model_type='lateral', device=device)
    
    t_span = (0, 2.0)
    delta_input = 0.1
    y0_lat = [0.0, 0.0]
    
    # 1. Generate Ground Truth
    t_true, y_true = vp_lat.generate_data(t_span, y0_lat, delta_input, n_points=500)
    
    # 2. Train (Hybrid Mode)
    t_col = np.random.uniform(t_span[0], t_span[1], 1000)
    
    # Sparse data
    idx_sparse = np.linspace(0, len(t_true)-1, 10, dtype=int)
    t_sparse = t_true[idx_sparse]
    y_sparse = y_true[idx_sparse]
    
    vp_lat.train(mode='hybrid', t_train=t_sparse, y_train=y_sparse, 
                 t_collocation=t_col, u=delta_input, epochs=1000)
    
    y_pred = vp_lat.predict(t_true)
    vp_lat.visualize(t_true, y_true, t_true, y_pred, title_suffix="(Lateral)")
    plt.savefig('result_lateral.png')
    print("Saved result_lateral.png")

    # =========================================================
    # Scenario 2: Longitudinal Model
    # =========================================================
    print("\n================================================")
    print("Scenario 2: Longitudinal Dynamics (Velocity)")
    print("================================================")
    
    vp_long = VehiclePINN(model_type='longitudinal', device=device)
    
    y0_long = [0.0] # Start from rest
    Fx_input = 3000.0 # Constant driving force
    
    t_true, y_true = vp_long.generate_data(t_span, y0_long, Fx_input, n_points=500)
    
    idx_sparse = np.linspace(0, len(t_true)-1, 10, dtype=int)
    t_sparse = t_true[idx_sparse]
    y_sparse = y_true[idx_sparse]
    
    vp_long.train(mode='hybrid', t_train=t_sparse, y_train=y_sparse,
                  t_collocation=t_col, u=Fx_input, epochs=1000)
                  
    y_pred = vp_long.predict(t_true)
    vp_long.visualize(t_true, y_true, t_true, y_pred, title_suffix="(Longitudinal)")
    plt.savefig('result_longitudinal.png')
    print("Saved result_longitudinal.png")
    
    # =========================================================
    # Scenario 3: Combined Model (Lat + Long)
    # =========================================================
    print("\n================================================")
    print("Scenario 3: Combined Dynamics (v, Beta, r)")
    print("================================================")
    
    vp_comb = VehiclePINN(model_type='combined', device=device)
    
    # Initial: v=10, beta=0, r=0
    y0_comb = [10.0, 0.0, 0.0] 
    # Inputs: Fx=2000, delta=0.1
    u_comb = [2000.0, 0.1] 
    
    t_true, y_true = vp_comb.generate_data(t_span, y0_comb, u_comb, n_points=500)
    
    idx_sparse = np.linspace(0, len(t_true)-1, 20, dtype=int)
    t_sparse = t_true[idx_sparse]
    y_sparse = y_true[idx_sparse]
    
    # Pass u as list for training
    vp_comb.train(mode='hybrid', t_train=t_sparse, y_train=y_sparse,
                  t_collocation=t_col, u=u_comb, epochs=2000, 
                  data_weight=10.0, pde_weight=1.0)
                  
    y_pred = vp_comb.predict(t_true)
    vp_comb.visualize(t_true, y_true, t_true, y_pred, title_suffix="(Combined)")
    plt.savefig('result_combined.png')
    print("Saved result_combined.png")

if __name__ == "__main__":
    main()
