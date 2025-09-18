import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import time

# ---------- 物理及网络参数 ----------
R_interface = 0.5       # 圆界半径
R2 = R_interface ** 2
k_out = 1e8            # 外部常数扩散系数（大跳量，可调）
np.random.seed(42)
N = 500
N_perturb = 1000
gamma_bc = 1000
gamma_interface = 1e6

# 隐层参数初始化
w_out = 1.0 * np.random.randn(N, 2)
b_out = 0.1 * np.random.randn(N)
w_inner = 1.0 * np.random.randn(N, 2)
b_inner = 0.1 * np.random.randn(N)

w_out1 = 7 * np.pi * np.random.randn(N_perturb, 2)
w_inner1 = 7 * np.pi * np.random.randn(N_perturb, 2)
b_out1 = 1.0 * np.random.randn(N_perturb)
b_inner1 = 1.0 * np.random.randn(N_perturb)
gamma = 100

# ---------- 采样点 ----------
def generate_points(N_sample, N_interface, Mb):
    x = np.linspace(-1, 1, N_sample)
    y = np.linspace(-1, 1, N_sample)
    xx, yy = np.meshgrid(x, y)
    index = np.where(xx ** 2 + yy ** 2 >= R2)
    external_points = np.column_stack((xx[index], yy[index]))
    index = np.where(xx ** 2 + yy ** 2 < R2)
    internal_points = np.column_stack((xx[index], yy[index]))

    theta = np.linspace(0, 2 * np.pi, N_interface)
    interface_points = np.column_stack((R_interface * np.cos(theta), R_interface * np.sin(theta)))
    m = Mb // 8
    left = np.column_stack([-np.ones(m), np.linspace(-1, 1, m)])
    right = np.column_stack([np.ones(m), np.linspace(-1, 1, m)])
    bottom = np.column_stack([np.linspace(-1.0, 1.0, m), -np.ones(m)])
    top = np.column_stack([np.linspace(-1.0, 1.0, m), np.ones(m)])
    boundary_points = np.concatenate([left, right, bottom, top], axis=0)
    return external_points, internal_points, interface_points, boundary_points

# ---------- 精确解与右端 ----------
def u_out_exact(x, y):
    return (x ** 3 + y ** 3) / k_out

def u_inner_exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def u_inner_grad_exact(x, y):
    ux = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    uy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    return ux, uy

def u_inner_laplace_exact(x, y):
    S = np.sin(np.pi * x) * np.sin(np.pi * y)
    return -2.0 * np.pi ** 2 * S

def f_out_rhs(x, y):
    return -(6.0 * x + 6.0 * y)

def f_inner_rhs(x, y):
    S = np.sin(np.pi * x) * np.sin(np.pi * y)
    beta = 1.0 + S**3
    lap = -2.0 * np.pi ** 2 * S
    grad2 = np.pi **2 * (np.cos(np.pi * x)**2 * np.sin(np.pi * y)**2 + np.sin(np.pi * x)**2 * np.cos(np.pi * y)**2)
    return -beta * lap - 3.0 * S ** 2 * grad2

def v(x, y):
    # 外部通量: k_out * ∇u_out·(x,y) = 3(x^3 + y^3)
    S = np.sin(np.pi * x) * np.sin(np.pi * y)
    flux_out = 3.0 * (x ** 3 + y ** 3)
    flux_in = (1.0 + S ** 3) * np.pi * (x * np.cos(np.pi * x) * np.sin(np.pi * y) +
                                        y * np.sin(np.pi * x) * np.cos(np.pi * y))
    return flux_out - flux_in

# ---------- 主迭代残差与雅可比 ----------
def compute_residual_and_jacobian(alpha_out, alpha_inner, external_points, internal_points, interface_points, boundary_points):
    M_out = len(external_points)
    M_inner = len(internal_points)
    M_gamma = len(interface_points)
    M_boundary = len(boundary_points)
    M_total = M_out + M_inner + M_gamma * 2 + M_boundary
    R = np.zeros(M_total)
    J = np.zeros((M_total, 2 * N))

    # 外部区
    for j in range(M_out):
        x, y = external_points[j]
        z_out = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        tanh_z_out = np.tanh(z_out)
        tanh_z_out_laplace = -2 * tanh_z_out * (1 - tanh_z_out ** 2)
        laplace_u_out = np.dot(alpha_out, (w_out[:, 0] ** 2 + w_out[:, 1] ** 2) * tanh_z_out_laplace)
        R[j] = k_out * laplace_u_out/k_out + f_out_rhs(x, y)/k_out
        J[j, :N] = k_out * (w_out[:, 0] ** 2 + w_out[:, 1] ** 2) * tanh_z_out_laplace/k_out

    # 内部区
    for j in range(M_inner):
        x, y = internal_points[j]
        z_inner = w_inner[:, 0] * x + w_inner[:, 1] * y + b_inner
        tanh_z_inner = np.tanh(z_inner)
        tanh_z_inner_deriv = 1 - tanh_z_inner ** 2
        tanh_z_inner_laplace = -2 * tanh_z_inner * tanh_z_inner_deriv
        u_inner = np.dot(alpha_inner, tanh_z_inner)
        laplace_u_inner = np.dot(alpha_inner, (w_inner[:, 0] ** 2 + w_inner[:, 1] ** 2) * tanh_z_inner_laplace)
        grad_u_inner_x = np.dot(alpha_inner, w_inner[:, 0] * tanh_z_inner_deriv)
        grad_u_inner_y = np.dot(alpha_inner, w_inner[:, 1] * tanh_z_inner_deriv)
        grad_inner_squared = grad_u_inner_x ** 2 + grad_u_inner_y ** 2
        beta_inner = 1.0 + u_inner ** 3
        beta_inner_u = 3.0 * u_inner ** 2
        beta_inner_uu = 6.0 * u_inner
        f_in_val = f_inner_rhs(x, y)
        R[j + M_out] = beta_inner * laplace_u_inner + beta_inner_u * grad_inner_squared + f_in_val
        # 雅可比
        term1_part1 = beta_inner_u * laplace_u_inner * tanh_z_inner
        term1_part2 = beta_inner * (w_inner[:, 0] ** 2 + w_inner[:, 1] ** 2) * tanh_z_inner_laplace
        term2_part1 = beta_inner_uu * grad_inner_squared * tanh_z_inner
        term2_part2 = 2.0 * beta_inner_u * (grad_u_inner_x * w_inner[:, 0] * tanh_z_inner_deriv + grad_u_inner_y * w_inner[:, 1] * tanh_z_inner_deriv)
        J[j + M_out, N:] = term1_part1 + term1_part2 + term2_part1 + term2_part2

    # 界面
    for j in range(M_gamma):
        x, y = interface_points[j]
        z_out = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        tanh_z_out = np.tanh(z_out)
        tanh_z_out_deriv = 1 - tanh_z_out ** 2
        u_out = np.dot(alpha_out, tanh_z_out)
        grad_u_out_x = np.dot(alpha_out, w_out[:, 0] * tanh_z_out_deriv)
        grad_u_out_y = np.dot(alpha_out, w_out[:, 1] * tanh_z_out_deriv)
        z_inner = w_inner[:, 0] * x + w_inner[:, 1] * y + b_inner
        tanh_z_inner = np.tanh(z_inner)
        tanh_z_inner_deriv = 1 - tanh_z_inner ** 2
        u_inner = np.dot(alpha_inner, tanh_z_inner)
        grad_u_inner_x = np.dot(alpha_inner, w_inner[:, 0] * tanh_z_inner_deriv)
        grad_u_inner_y = np.dot(alpha_inner, w_inner[:, 1] * tanh_z_inner_deriv)
        beta_out = k_out
        beta_inner = 1.0 + u_inner ** 3
        beta_inner_u = 3.0 * u_inner ** 2
        # 连续性
        R_gamma_v = (u_out - u_inner - (u_out_exact(x, y) - u_inner_exact(x, y)))
        R[j + M_out + M_inner] = R_gamma_v * gamma_interface
        J[j + M_out + M_inner, :N] = tanh_z_out * gamma_interface
        J[j + M_out + M_inner, N:] = -tanh_z_inner * gamma_interface
        # 通量
        flux_jump = (beta_out * (grad_u_out_x * x + grad_u_out_y * y) - beta_inner * (grad_u_inner_x * x + grad_u_inner_y * y)) - v(x, y)
        R[j + M_out + M_inner + M_gamma] = flux_jump/beta_out * gamma_interface
        J[j + M_out + M_inner + M_gamma, :N] = beta_out * (w_out[:, 0] * tanh_z_out_deriv * x + w_out[:, 1] * tanh_z_out_deriv * y)/beta_out * gamma_interface
        term_inner = (beta_inner_u * tanh_z_inner * (grad_u_inner_x * x + grad_u_inner_y * y)) + (beta_inner * (w_inner[:, 0] * tanh_z_inner_deriv * x + w_inner[:, 1] * tanh_z_inner_deriv * y))
        J[j + M_out + M_inner + M_gamma, N:] = -term_inner/beta_out * gamma_interface

    # 边界
    for j in range(M_boundary):
        x, y = boundary_points[j]
        z_out = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        tanh_z_out = np.tanh(z_out)
        u_out = np.dot(alpha_out, tanh_z_out)
        R_bc1 = (u_out - u_out_exact(x, y))
        R[j + M_out + M_inner + M_gamma * 2] = R_bc1 * gamma_bc
        J[j + M_out + M_inner + M_gamma * 2, :N] = tanh_z_out * gamma_bc

    return R, J

# ---------- 摄动解残差与雅可比（与主迭代结构对应） ----------
def compute_perturb_residual(alpha_out_p, alpha_inner_p, alpha_out, alpha_inner, X_out, X_inner, X_gamma, X_omega):
    X = np.vstack([X_out, X_inner, X_gamma, X_gamma, X_omega])
    M_out, M_inner = len(X_out), len(X_inner)
    M_gamma = len(X_gamma)
    M_omega = len(X_omega)
    M_total = M_out + M_inner + 2 * M_gamma + M_omega
    R = np.zeros(M_total)
    J = np.zeros((M_total, 2 * N_perturb))

    for j in range(M_total):
        x, y = X[j]
        # 主解
        z_out = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        tanh_z_out = np.tanh(z_out)
        tanh_z_out_deriv = 1 - tanh_z_out ** 2
        tanh_z_out_laplace = -2 * tanh_z_out * tanh_z_out_deriv
        u_out = np.dot(alpha_out, tanh_z_out)
        grad_u_out_x = np.dot(alpha_out, w_out[:, 0] * tanh_z_out_deriv)
        grad_u_out_y = np.dot(alpha_out, w_out[:, 1] * tanh_z_out_deriv)
        laplace_u_out = np.dot(alpha_out, (w_out[:, 0]**2 + w_out[:, 1]**2) * tanh_z_out_laplace)
        z_inner = w_inner[:, 0] * x + w_inner[:, 1] * y + b_inner
        tanh_z_inner = np.tanh(z_inner)
        tanh_z_inner_deriv = 1 - tanh_z_inner ** 2
        tanh_z_inner_laplace = -2 * tanh_z_inner * tanh_z_inner_deriv
        u_inner = np.dot(alpha_inner, tanh_z_inner)
        grad_u_inner_x = np.dot(alpha_inner, w_inner[:, 0] * tanh_z_inner_deriv)
        grad_u_inner_y = np.dot(alpha_inner, w_inner[:, 1] * tanh_z_inner_deriv)
        laplace_u_inner = np.dot(alpha_inner, (w_inner[:, 0]**2 + w_inner[:, 1]**2) * tanh_z_inner_laplace)
        # 摄动
        z_out_p = w_out1[:, 0] * x + w_out1[:, 1] * y + b_out1
        cos_z_out_p = np.cos(z_out_p)
        sin_z_out_p = np.sin(z_out_p)
        u_out_p = np.dot(alpha_out_p, sin_z_out_p)
        grad_u_out_px = np.dot(alpha_out_p, w_out1[:, 0] * cos_z_out_p)
        grad_u_out_py = np.dot(alpha_out_p, w_out1[:, 1] * cos_z_out_p)
        laplace_u_out_p = np.dot(alpha_out_p, (w_out1[:, 0]**2 + w_out1[:, 1]**2) * -sin_z_out_p)
        z_inner_p = w_inner1[:, 0] * x + w_inner1[:, 1] * y + b_inner1
        cos_z_inner_p = np.cos(z_inner_p)
        sin_z_inner_p = np.sin(z_inner_p)
        u_inner_p = np.dot(alpha_inner_p, sin_z_inner_p)
        grad_u_inner_px = np.dot(alpha_inner_p, w_inner1[:, 0] * cos_z_inner_p)
        grad_u_inner_py = np.dot(alpha_inner_p, w_inner1[:, 1] * cos_z_inner_p)
        laplace_u_inner_p = np.dot(alpha_inner_p, (w_inner1[:, 0]**2 + w_inner1[:, 1]**2) * -sin_z_inner_p)
        # 区域与条件划分
        if j < M_out + M_inner:
            if j < M_out:
                R_out = (k_out * laplace_u_out + f_out_rhs(x, y)) / epsilon + k_out * laplace_u_out_p
                R[j] = R_out/k_out
                J[j, :N_perturb] = k_out * (w_out1[:, 0]**2 + w_out1[:, 1]**2) * (-sin_z_out_p)/k_out
            else:
                beta_in = 1.0 + u_inner ** 3
                beta_in_u = 3.0 * u_inner ** 2
                beta_in_uu = 6.0 * u_inner
                beta_in_uuu = 6.0
                f_in_val = f_inner_rhs(x, y)
                R_in = (beta_in_u * (grad_u_inner_x ** 2 + grad_u_inner_y ** 2) + beta_in * laplace_u_inner + f_in_val) / epsilon + \
                    2.0 * beta_in_u * (grad_u_inner_x * grad_u_inner_px + grad_u_inner_y * grad_u_inner_py) + \
                    beta_in_uu * (grad_u_inner_x ** 2 + grad_u_inner_y ** 2) * u_inner_p + beta_in_u * u_inner_p * laplace_u_inner + \
                    beta_in * laplace_u_inner_p + \
                    epsilon * (2.0 * u_inner_p * beta_in_uu * (grad_u_inner_x * grad_u_inner_px + grad_u_inner_y * grad_u_inner_py) + \
                               0.5 * u_inner_p ** 2 * beta_in_uuu * (grad_u_inner_x ** 2 + grad_u_inner_y ** 2) + 0.5 * u_inner_p ** 2 * beta_in_uu * laplace_u_inner + \
                               beta_in_u * (grad_u_inner_px ** 2 + grad_u_inner_py ** 2) + u_inner_p * beta_in_u * laplace_u_inner_p)
                R[j] = R_in
                # 雅可比
                J_inner = 2.0 * beta_in_u * (grad_u_inner_x * w_inner1[:, 0] * cos_z_inner_p + grad_u_inner_y * w_inner1[:, 1] * cos_z_inner_p) + \
                          beta_in_uu * (grad_u_inner_x ** 2 + grad_u_inner_y ** 2) * sin_z_inner_p + beta_in_u * laplace_u_inner * sin_z_inner_p + \
                          beta_in * (w_inner1[:, 0] ** 2 + w_inner1[:, 1] ** 2) * (-sin_z_inner_p) + \
                          epsilon * (2.0 * sin_z_inner_p * beta_in_uu * (grad_u_inner_x * grad_u_inner_px + grad_u_inner_y * grad_u_inner_py) + \
                                     2.0 * u_inner_p * beta_in_uu * (grad_u_inner_x * w_inner1[:, 0] + grad_u_inner_y * w_inner1[:, 1]) * cos_z_inner_p + \
                                     sin_z_inner_p * u_inner_p * beta_in_uuu * (grad_u_inner_x ** 2 + grad_u_inner_y ** 2) + sin_z_inner_p * u_inner_p * beta_in_uu * laplace_u_inner + \
                                     2.0 * beta_in_u * (grad_u_inner_px * w_inner1[:, 0] * cos_z_inner_p + grad_u_inner_py * w_inner1[:, 1] * cos_z_inner_p) + \
                                     beta_in_u * laplace_u_inner_p * sin_z_inner_p + beta_in_u * u_inner_p * (w_inner1[:, 0] ** 2 + w_inner1[:, 1] ** 2) * -sin_z_inner_p)
                J[j, N_perturb:] = J_inner
        else:
            gamma_idx = j - (M_out + M_inner)
            if gamma_idx < 2 * M_gamma:
                if gamma_idx < M_gamma:
                    R_gamn = (u_out - u_inner - (u_out_exact(x, y) - u_inner_exact(x, y))) / epsilon + (u_out_p - u_inner_p)
                    R[j] = R_gamn * gamma
                    J[j, :N_perturb] = sin_z_out_p * gamma
                    J[j, N_perturb:] = -sin_z_inner_p * gamma
                else:
                    beta_in = 1.0 + u_inner ** 3
                    beta_in_u = 3.0 * u_inner ** 2
                    beta_in_uu = 6.0 * u_inner
                    R_gammad = (k_out * (grad_u_out_x * x + grad_u_out_y * y) - beta_in * (grad_u_inner_x * x + grad_u_inner_y * y) - v(x, y)) / epsilon + \
                        (k_out * (grad_u_out_px * x + grad_u_out_py * y) - beta_in_u * u_inner_p * (grad_u_inner_x * x + grad_u_inner_y * y) - beta_in * (grad_u_inner_px * x + grad_u_inner_py * y)) + \
                        epsilon * ( -0.5 * u_inner_p**2 * beta_in_uu * (grad_u_inner_x * x + grad_u_inner_y * y) - u_inner_p * beta_in_u * (grad_u_inner_px * x + grad_u_inner_py * y) )
                    R[j] = R_gammad * gamma
                    # 雅可比
                    J[j, :N_perturb] = k_out * (w_out1[:, 0] * x + w_out1[:, 1] * y) * cos_z_out_p * gamma
                    J_gammad_inner = -((beta_in_u * sin_z_inner_p * (grad_u_inner_x * x + grad_u_inner_y * y) + beta_in * (w_inner1[:, 0] * x + w_inner1[:, 1] * y) * cos_z_inner_p) + \
                        epsilon * (u_inner_p * sin_z_inner_p * beta_in_uu * (grad_u_inner_x * x + grad_u_inner_y * y) + sin_z_inner_p * beta_in_u * (grad_u_inner_px * x + grad_u_inner_py * y) + \
                                   u_inner_p * beta_in_u * (w_inner1[:, 0] * x + w_inner1[:, 1] * y) * cos_z_inner_p))
                    J[j, N_perturb:] = J_gammad_inner * gamma
            else:
                R_bc1 = (u_out + u_out_p * epsilon - u_out_exact(x, y)) / epsilon
                R[j] = R_bc1 * gamma * 10
                J[j, :N_perturb] = sin_z_out_p * gamma * 10
    return R, J

# ---------- 测试网格与真解 ----------
N_test = 501
x_test = np.linspace(-1, 1, N_test)
y_test = np.linspace(-1, 1, N_test)
xx, yy = np.meshgrid(x_test, y_test)
points = np.column_stack((xx.ravel(), yy.ravel()))
is_external = points[:, 0] ** 2 + points[:, 1] ** 2 >= R2
external_points_test = points[is_external]
internal_points_test = points[~is_external]

u_exact = np.where(xx ** 2 + yy ** 2 >= R2, u_out_exact(xx, yy), u_inner_exact(xx, yy))
u_exact_norm = np.linalg.norm(u_exact.ravel())

def predict(points, w, b, alpha):
    z = np.dot(points, w.T) + b
    return np.dot(np.tanh(z), alpha)

def predict1(points, w, b, alpha):
    z = np.dot(points, w.T) + b
    return np.dot(np.sin(z), alpha)

# ---------- 主解牛顿迭代 ----------
alpha_out = np.zeros(N)
alpha_inner = np.zeros(N)
tol = 1e-10
max_iter = 50
delta_threshold = 1e-4
prev_residual = float('inf')
main_residual_history = []
main_l2_history = []
external_points, internal_points, interface_points, boundary_points = generate_points(101, 101, 404)

for k in range(max_iter):
    R, J = compute_residual_and_jacobian(alpha_out, alpha_inner, external_points, internal_points, interface_points, boundary_points)
    M_total = len(R)
    residual_norm = np.linalg.norm(R) / np.sqrt(M_total)
    main_residual_history.append(residual_norm)
    print(f"Iter {k}: Residual norm (RMSE) = {residual_norm:.4e}")
    u_pred = np.zeros(len(points))
    u_pred[is_external] = predict(external_points_test, w_out, b_out, alpha_out)
    u_pred[~is_external] = predict(internal_points_test, w_inner, b_inner, alpha_inner)
    u_pred = u_pred.reshape(xx.shape)
    error = u_exact - u_pred
    rel_l2 = np.linalg.norm(error.ravel()) / u_exact_norm
    main_l2_history.append(rel_l2)
    residual_diff = abs(residual_norm - prev_residual) / abs(residual_norm) if residual_norm != 0 else float('inf')
    if residual_diff < delta_threshold:
        print(f"Main iteration stopped: residual difference {residual_diff:.4e} < {delta_threshold}")
        break
    prev_residual = residual_norm
    if residual_norm < tol:
        break
    delta_beta, _, _, _ = lstsq(J, -R, cond=1e-12)
    alpha_out += delta_beta[:N]
    alpha_inner += delta_beta[N:]

u_pred_main = np.zeros(len(points))
u_pred_main[is_external] = predict(external_points_test, w_out, b_out, alpha_out)
u_pred_main[~is_external] = predict(internal_points_test, w_inner, b_inner, alpha_inner)
u_pred_main = u_pred_main.reshape(xx.shape)

# ---------- 摄动阶段 ----------
perturb_residual_history = []
perturb_l2_history = []
epsilon = 1e-4
alpha_out_p = np.zeros(N_perturb)
alpha_inner_p = np.zeros(N_perturb)
max_perturb_iter = 50
perturb_tol = 1e-10
external_points_p, internal_points_p, interface_points_p, boundary_points_p = generate_points(126, 126, 504)
prev_perturb_residual = float('inf')
start_time = time.time()

for k in range(max_perturb_iter):
    R_p, J_p = compute_perturb_residual(alpha_out_p, alpha_inner_p, alpha_out, alpha_inner, external_points_p, internal_points_p, interface_points_p, boundary_points_p)
    M_total_p = len(R_p)
    res_norm = np.linalg.norm(R_p) / np.sqrt(M_total_p)
    print(f"Perturb Iter {k}: Residual (RMSE) = {res_norm:.3e}")
    perturb_residual_history.append(res_norm)
    u_pred1_current = np.zeros(len(points))
    u_pred1_current[is_external] = predict1(external_points_test, w_out1, b_out1, alpha_out_p)
    u_pred1_current[~is_external] = predict1(internal_points_test, w_inner1, b_inner1, alpha_inner_p)
    u_pred1_current = u_pred1_current.reshape(xx.shape)
    u_total = u_pred_main + epsilon * u_pred1_current
    error_total = u_exact - u_total
    rel_l2_total = np.linalg.norm(error_total.ravel()) / u_exact_norm
    perturb_l2_history.append(rel_l2_total)
    residual_diff = abs(res_norm - prev_perturb_residual)
    if residual_diff < delta_threshold:
        print(f"Perturb iteration stopped: residual difference {residual_diff:.4e} < {delta_threshold}")
        break
    prev_perturb_residual = res_norm
    if res_norm < perturb_tol:
        break
    if res_norm>1e-1:
        cond0 = res_norm * 1e-14
    else:
        cond0 = 0.0
    delta_p, _, _, _ = lstsq(J_p, -R_p, cond=1e-14)
    alpha_out_p += delta_p[:N_perturb]
    alpha_inner_p += delta_p[N_perturb:]

end_time = time.time()
elapsed = end_time - start_time
print(f"Elapsed time: {elapsed:.6f} seconds")

# ---------- 最终预测&误差 ----------
u_pred = np.zeros(len(points))
u_pred[is_external] = predict(external_points_test, w_out, b_out, alpha_out)
u_pred[~is_external] = predict(internal_points_test, w_inner, b_inner, alpha_inner)
u_pred = u_pred.reshape(xx.shape)

u_pred1 = np.zeros(len(points))
u_pred1[is_external] = predict1(external_points_test, w_out1, b_out1, alpha_out_p)
u_pred1[~is_external] = predict1(internal_points_test, w_inner1, b_inner1, alpha_inner_p)
u_pred1 = u_pred1.reshape(xx.shape)

error = u_exact - u_pred
error1 = u_exact - (u_pred + epsilon * u_pred1)

max_abs_error = np.max(np.abs(error))
rel_l2_error = np.linalg.norm(error.ravel()) / u_exact_norm
rel_l2_error_stage2 = np.linalg.norm(error1.ravel()) / u_exact_norm
rel_max_error_stage2 = np.max(np.abs(error1)) / np.max(np.abs(u_exact))

print(f"Max absolute error (stage 1): {max_abs_error:.4e}")
print(f"Relative L2 error (stage 1): {rel_l2_error:.4e}")
print(f"Relative L2 error (stage 2): {rel_l2_error_stage2:.4e}")
print(f"Relative max error (stage 2): {rel_max_error_stage2:.4e}")