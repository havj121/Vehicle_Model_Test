import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------
# 1. 车辆参数
# ------------------------------
m = 1500.0          # 质量 (kg)
Iz = 3000.0         # 横摆转动惯量 (kg*m^2)
a = 1.2             # 质心到前轴距离 (m)
b = 1.5             # 质心到后轴距离 (m)
Cf = 80000.0        # 前轮侧偏刚度 (N/rad)
Cr = 80000.0        # 后轮侧偏刚度 (N/rad)
Vx = 20.0           # 纵向速度 (m/s)
delta = 0.1         # 前轮转角 (rad), 阶跃输入

# 状态空间系数
a11 = -(Cf + Cr) / (m * Vx)
a12 = -1.0 + (-a * Cf + b * Cr) / (m * Vx**2)
a21 = -(a * Cf - b * Cr) / Iz
a22 = -(a**2 * Cf + b**2 * Cr) / (Iz * Vx)
b1 = Cf / (m * Vx)
b2 = a * Cf / Iz

# 初始条件
beta0 = 0.0
r0 = 0.0

# 时间区间
t_min, t_max = 0.0, 5.0

# ------------------------------
# 2. 生成参考数值解（用于对比，不参与训练）
# ------------------------------
def vehicle_ode(t, y):
    beta, r = y
    dbeta_dt = a11 * beta + a12 * r + b1 * delta
    dr_dt = a21 * beta + a22 * r + b2 * delta
    return [dbeta_dt, dr_dt]

sol = solve_ivp(vehicle_ode, (t_min, t_max), [beta0, r0], t_eval=np.linspace(t_min, t_max,2000))
t_ref = sol.t
beta_ref = sol.y[0]
r_ref = sol.y[1]

# ------------------------------
# 3. PINN 网络定义
# ------------------------------
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList()
        for i in range(len(layers)-1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
        # Xavier初始化
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, t):
        # t: [batch, 1]
        x = t
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)  # 最后一层无激活，输出 [beta, r]
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pinn = PINN(layers=[1, 64, 64, 2]).to(device)

# ------------------------------
# 4. 损失函数
# ------------------------------
def pde_residual(t):
    """计算给定时间点 t 的 PDE 残差"""
    t.requires_grad = True
    out = pinn(t)               # [batch, 2]
    beta = out[:, 0:1]
    r = out[:, 1:2]

    # 自动微分求时间导数
    beta_t = torch.autograd.grad(beta, t, grad_outputs=torch.ones_like(beta), create_graph=True)[0]
    r_t = torch.autograd.grad(r, t, grad_outputs=torch.ones_like(r), create_graph=True)[0]

    # 物理方程右端项
    dbeta_dt_pred = a11 * beta + a12 * r + b1 * delta
    dr_dt_pred = a21 * beta + a22 * r + b2 * delta

    # 残差
    loss_beta = torch.mean((beta_t - dbeta_dt_pred) ** 2)
    loss_r = torch.mean((r_t - dr_dt_pred) ** 2)
    return loss_beta + loss_r

def initial_loss(t0):
    """计算初始条件损失"""
    out0 = pinn(t0)
    beta0_pred = out0[:, 0:1]
    r0_pred = out0[:, 1:2]
    loss_beta0 = torch.mean((beta0_pred - beta0) ** 2)
    loss_r0 = torch.mean((r0_pred - r0) ** 2)
    return loss_beta0 + loss_r0

# ------------------------------
# 5. 训练
# ------------------------------
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
epochs = 4000
N_pde = 1000          # 每个 epoch 随机采样点数
t_max_train = 3.0
t_min_train = 1.0


t0_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(device)

loss_history = []

for epoch in range(epochs):
    # 随机采样时间点用于 PDE 损失
    t_pde = np.random.uniform(t_min_train, t_max_train, (N_pde, 1))
    t_pde_tensor = torch.tensor(t_pde, dtype=torch.float32, requires_grad=True).to(device)

    optimizer.zero_grad()

    loss_ic = initial_loss(t0_tensor)
    loss_pde = pde_residual(t_pde_tensor)
    loss = loss_ic + loss_pde

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 500 == 0:
        print(f'Epoch {epoch:5d}, Loss: {loss.item():.2e}, IC: {loss_ic.item():.2e}, PDE: {loss_pde.item():.2e}')

# ------------------------------
# 6. 结果可视化
# ------------------------------
# 预测
t_test = np.linspace(t_min, t_max, 500).reshape(-1, 1)
t_test_tensor = torch.tensor(t_test, dtype=torch.float32).to(device)
with torch.no_grad():
    pred = pinn(t_test_tensor).cpu().numpy()
beta_pred = pred[:, 0]
r_pred = pred[:, 1]

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t_ref, beta_ref, 'b-', label='Numerical')
plt.plot(t_test, beta_pred, 'r--', label='PINN')
plt.xlabel('t (s)')
plt.ylabel('β (rad)')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(t_ref, r_ref, 'b-', label='Numerical')
plt.plot(t_test, r_pred, 'r--', label='PINN')
plt.xlabel('t (s)')
plt.ylabel('r (rad/s)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# 绘制损失曲线
plt.figure()
plt.semilogy(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid()
plt.show()