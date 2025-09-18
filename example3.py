import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import time

# ------------------------- 基本参数与几何设置 -------------------------
k_val = 1

# 梅花（五瓣花）界面参数：r(θ) = R0 + A * cos(m * θ)
R0 = 0.5       # 基础半径（原圆半径）
A_plum = 0.10  # 花形振幅，需保证R(θ) > 0
m_petal = 5    # 梅花瓣数，五瓣

# ------------------------- 梅花曲线及法向计算 -------------------------
def R_theta(theta):
    # r(θ)
    return R0 + A_plum * np.cos(m_petal * theta)

def dR_dtheta(theta):
    # r'(θ)
    return -A_plum * m_petal * np.sin(m_petal * theta)

def phi_levelset(x, y):
    # 水平集函数：φ(x,y) = r - R(θ)，φ<0为内部，φ>0为外部
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return r - R_theta(theta)

def normal_vec(x, y):
    # 计算梅花界面上的单位外法向（从内域指向外域）
    # 利用φ(x,y)=r-R(θ)，n = ∇φ / |∇φ|
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    # Rt = R_theta(theta) # 此行无用
    dRt = dR_dtheta(theta)
    # 防止除零
    if r < 1e-14:
        return 1.0, 0.0
    # ∇φ = (∂r/∂x - R'(θ) ∂θ/∂x, ∂r/∂y - R'(θ) ∂θ/∂y)
    # ∂r/∂x = x/r, ∂r/∂y = y/r; ∂θ/∂x = -y/r^2, ∂θ/∂y = x/r^2
    gx = x / r - dRt * (-y / r**2)
    gy = y / r - dRt * ( x / r**2)
    gnorm = np.hypot(gx, gy)
    if gnorm < 1e-14:
        return 1.0, 0.0
    return gx / gnorm, gy / gnorm

# ------------------------- 点集生成（外部/内部/界面/边界） -------------------------
def generate_points(N_sample, N_interface, Mb):
    # 采样方形区域[-1,1]^2
    x = np.linspace(-1, 1, N_sample)
    y = np.linspace(-1, 1, N_sample)
    xx, yy = np.meshgrid(x, y)
    rr = np.hypot(xx, yy)
    theta = np.arctan2(yy, xx)
    Rg = R_theta(theta)
    # 外部/内部点的划分基于φ=r-R(θ)
    mask_external = rr >= Rg
    external_x = xx[mask_external]
    external_y = yy[mask_external]
    external_points = np.column_stack((external_x, external_y))

    mask_internal = ~mask_external
    internal_x = xx[mask_internal]
    internal_y = yy[mask_internal]
    internal_points = np.column_stack((internal_x, internal_y))

    # 界面参数点（按θ等分）
    theta_i = np.linspace(0, 2*np.pi, N_interface, endpoint=True)
    Ri = R_theta(theta_i)
    interface_points = np.column_stack((Ri * np.cos(theta_i), Ri * np.sin(theta_i)))

    # 方形外边界采样
    m = Mb // 8  # 每边点数
    left = np.column_stack([-np.ones(m), np.linspace(-1, 1, m)])
    right = np.column_stack([np.ones(m), np.linspace(-1, 1, m)])
    bottom = np.column_stack([np.linspace(-1.0, 1.0, m), -np.ones(m)])
    top = np.column_stack([np.linspace(-1.0, 1.0, m), np.ones(m)])
    boundary_points = np.concatenate([left, right, bottom, top], axis=0)

    return external_points, internal_points, interface_points, boundary_points

# ------------------------- 精确解与所需函数 -------------------------
def u_out_exact(x, y):
    return 0.5 * np.sin(k_val * np.pi * x) * np.sin(k_val * np.pi * y) + 0.25

def u_inner_exact(x, y):
    return 0.25 - (x**2 + y**2)

def grad_u_out_exact(x, y):
    # 外域精确解梯度
    ux = 0.5 * k_val * np.pi * np.cos(k_val * np.pi * x) * np.sin(k_val * np.pi * y)
    uy = 0.5 * k_val * np.pi * np.sin(k_val * np.pi * x) * np.cos(k_val * np.pi * y)
    return ux, uy

def grad_u_inner_exact(x, y):
    # 内域精确解梯度
    return -2.0 * x, -2.0 * y

def v(x, y):
    # 第二跳跃条件右端：通量跳跃在梅花界面上的值
    # v = beta_out(u_out_exact) * ∇u_out_exact · n_hat - beta_in(u_in_exact) * ∇u_in_exact · n_hat
    nx, ny = normal_vec(x, y)
    uo = u_out_exact(x, y)
    ui = u_inner_exact(x, y)
    ux_o, uy_o = grad_u_out_exact(x, y)
    ux_i, uy_i = grad_u_inner_exact(x, y)
    beta_out = x**2 + y**2 + np.exp(0.5 * uo)
    beta_in = 1.0 + np.sin(ui)
    flux_out = beta_out * (ux_o * nx + uy_o * ny)
    flux_in = beta_in * (ux_i * nx + uy_i * ny)
    return flux_out - flux_in

# ------------------------- 随机特征初始化（ELM） -------------------------
np.random.seed(42)
N = 500
N_perturb = 4000
gamma_bc = 1000
gamma_interface = 1000

w_out = 1.0 * np.random.randn(N, 2)
b_out = 0.1 * np.random.randn(N)
w_inner = 1.0 * np.random.randn(N, 2)
b_inner = 0.1 * np.random.randn(N)

w_out1 = 7*np.pi * np.random.randn(N_perturb, 2)
w_inner1 = 7*np.pi * np.random.randn(N_perturb, 2)
b_out1 = 1.0 * np.random.randn(N_perturb)
b_inner1 = 1.0 * np.random.randn(N_perturb)
gamma = 100

# ------------------------- 主问题残差与雅可比 -------------------------
def compute_residual_and_jacobian(alpha_out, alpha_inner, external_points, internal_points, interface_points, boundary_points):
    M_out = len(external_points)
    M_inner = len(internal_points)
    M_gamma = len(interface_points)
    M_boundary = len(boundary_points)

    M_total = M_out + M_inner + M_gamma*2 +  M_boundary
    R = np.zeros(M_total)
    J = np.zeros((M_total, 2 * N))

    # 外域内部点
    for j in range(M_out):
        x, y = external_points[j]
        z_out = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        tanh_z_out = np.tanh(z_out)
        tanh_z_out_deriv = 1 - tanh_z_out ** 2
        tanh_z_out_laplace = -2 * tanh_z_out * tanh_z_out_deriv

        u_out = np.dot(alpha_out, tanh_z_out)
        laplace_u_out = np.dot(alpha_out, (w_out[:, 0] ** 2 + w_out[:, 1] ** 2) * tanh_z_out_laplace)
        grad_u_out_x = np.dot(alpha_out, w_out[:, 0] * tanh_z_out_deriv)
        grad_u_out_y = np.dot(alpha_out, w_out[:, 1] * tanh_z_out_deriv)
        grad_out_squared = grad_u_out_x ** 2 + grad_u_out_y ** 2

        f_term1 = (k_val * np.pi) ** 2 * np.sin(k_val * np.pi * x) * np.sin(k_val * np.pi * y)
        f_term2 = (k_val * np.pi) ** 2 / 4 * (np.cos(k_val * np.pi * x) ** 2 * np.sin(k_val * np.pi * y) ** 2 + np.sin(
            k_val * np.pi * x) ** 2 * np.cos(k_val * np.pi * y) ** 2)
        grad_u_x = 0.5 * k_val * np.pi * np.cos(k_val * np.pi * x) * np.sin(k_val * np.pi * y)
        grad_u_y = 0.5 * k_val * np.pi * np.sin(k_val * np.pi * x) * np.cos(k_val * np.pi * y)
        f_out = (x ** 2 + y ** 2) * f_term1 + np.exp(0.5 * u_out_exact(x, y)) * f_term1 - 2 * (
                    x * grad_u_x + y * grad_u_y) - 0.5 * np.exp(0.5 * u_out_exact(x, y)) * f_term2

        f_out_u = 0.0
        beta_out = x**2 + y**2 + np.exp(0.5 * u_out)
        beta_x = 2 * x
        beta_y = 2 * y
        beta_out_u = 0.5 * np.exp(0.5 * u_out)
        beta_out_uu = 0.25 * np.exp(0.5 * u_out)

        R_term = beta_out * laplace_u_out + beta_out_u * grad_out_squared + (beta_x * grad_u_out_x + beta_y * grad_u_out_y) + f_out
        R[j] = R_term

        term1_part1 = beta_out_u * laplace_u_out * tanh_z_out
        term1_part2 = beta_out * (w_out[:, 0] ** 2 + w_out[:, 1] ** 2) * tanh_z_out_laplace
        term2_part1 = beta_out_uu * grad_out_squared * tanh_z_out
        term2_part2 = 2 * beta_out_u * (grad_u_out_x * w_out[:, 0] * tanh_z_out_deriv + grad_u_out_y * w_out[:, 1] * tanh_z_out_deriv)
        term3 = (beta_x * w_out[:, 0] * tanh_z_out_deriv + beta_y * w_out[:, 1] * tanh_z_out_deriv)
        J[j, :N] = (term1_part1 + term1_part2 + term2_part1 + term2_part2 + term3 + f_out_u * tanh_z_out)

    # 内域内部点
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

        f_inner = 4 * (np.sin(u_inner) + 1 - (x**2 + y**2) * np.cos(u_inner))
        f_inner_u = 4 * (np.cos(u_inner) + (x**2 + y**2) * np.sin(u_inner))

        beta_inner = 1 + np.sin(u_inner)
        beta_inner_u = np.cos(u_inner)
        beta_inner_uu = -np.sin(u_inner)

        R_term = beta_inner * laplace_u_inner + beta_inner_u * grad_inner_squared + f_inner
        R[j + M_out] = R_term

        term1_part1 = beta_inner_u * laplace_u_inner * tanh_z_inner
        term1_part2 = beta_inner * (w_inner[:, 0] ** 2 + w_inner[:, 1] ** 2) * tanh_z_inner_laplace
        term2_part1 = beta_inner_uu * grad_inner_squared * tanh_z_inner
        term2_part2 = 2 * beta_inner_u * (grad_u_inner_x * w_inner[:, 0] * tanh_z_inner_deriv + grad_u_inner_y * w_inner[:, 1] * tanh_z_inner_deriv)

        J[j + M_out, N:] = (term1_part1 + term1_part2 + term2_part1 + term2_part2 + f_inner_u * tanh_z_inner)

    # 界面条件（梅花曲线）
    for j in range(M_gamma):
        x, y = interface_points[j]
        nx, ny = normal_vec(x, y)

        # 外域
        z_out = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        tanh_z_out = np.tanh(z_out)
        tanh_z_out_deriv = 1 - tanh_z_out ** 2
        u_out = np.dot(alpha_out, tanh_z_out)
        grad_u_out_x = np.dot(alpha_out, w_out[:, 0] * tanh_z_out_deriv)
        grad_u_out_y = np.dot(alpha_out, w_out[:, 1] * tanh_z_out_deriv)
        beta_out = x**2 + y**2 + np.exp(0.5 * u_out)
        beta_out_u = 0.5 * np.exp(0.5 * u_out)

        # 内域
        z_inner = w_inner[:, 0] * x + w_inner[:, 1] * y + b_inner
        tanh_z_inner = np.tanh(z_inner)
        tanh_z_inner_deriv = 1 - tanh_z_inner ** 2
        u_inner = np.dot(alpha_inner, tanh_z_inner)
        grad_u_inner_x = np.dot(alpha_inner, w_inner[:, 0] * tanh_z_inner_deriv)
        grad_u_inner_y = np.dot(alpha_inner, w_inner[:, 1] * tanh_z_inner_deriv)
        beta_inner = 1 + np.sin(u_inner)
        beta_inner_u = np.cos(u_inner)

        # 第一跳跃条件：[u] = u_out - u_in - (u_out_exact - u_in_exact)
        R_gamman = (u_out - u_inner - (u_out_exact(x, y) - u_inner_exact(x, y)))
        R[j + M_out + M_inner] = R_gamman * gamma_interface
        J[j + M_out + M_inner, :N] = tanh_z_out * gamma_interface
        J[j + M_out + M_inner, N:] = -tanh_z_inner * gamma_interface

        # 第二跳跃条件：[β ∂u/∂n] = v(x,y)，其中n为梅花界面外法向
        gout_n = grad_u_out_x * nx + grad_u_out_y * ny
        ginn_n = grad_u_inner_x * nx + grad_u_inner_y * ny
        R_gammad = (beta_out * gout_n - beta_inner * ginn_n) - v(x, y)
        R[j + M_out + M_inner + M_gamma] = R_gammad * gamma_interface

        term_out = beta_out_u * tanh_z_out * gout_n + beta_out * (w_out[:, 0] * tanh_z_out_deriv * nx + w_out[:, 1] * tanh_z_out_deriv * ny)
        J[j + M_out + M_inner + M_gamma, :N] = term_out * gamma_interface

        term_inner = beta_inner_u * tanh_z_inner * ginn_n + beta_inner * (w_inner[:, 0] * tanh_z_inner_deriv * nx + w_inner[:, 1] * tanh_z_inner_deriv * ny)
        J[j + M_out + M_inner + M_gamma, N:] = -term_inner * gamma_interface

    # 外边界条件（方形）
    for j in range(M_boundary):
        x, y = boundary_points[j]
        z_out = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        tanh_z_out = np.tanh(z_out)
        u_out = np.dot(alpha_out, tanh_z_out)
        R_bc1 = (u_out - u_out_exact(x, y))

        R[j + M_out + M_inner + M_gamma * 2] = R_bc1 * gamma_bc
        J[j + M_out + M_inner + M_gamma * 2, :N] = tanh_z_out * gamma_bc

    return R, J

# ------------------------- 摄动子问题残差与雅可比 -------------------------
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
        # ================= 主解计算 =================
        z_out = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        tanh_z_out = np.tanh(z_out)
        tanh_z_out_deriv = 1 - tanh_z_out**2
        tanh_z_out_laplace = -2 * tanh_z_out * tanh_z_out_deriv
        u_out = np.dot(alpha_out, tanh_z_out)
        grad_u_out_x = np.dot(alpha_out, w_out[:, 0] * tanh_z_out_deriv)
        grad_u_out_y = np.dot(alpha_out, w_out[:, 1] * tanh_z_out_deriv)
        laplace_u_out = np.dot(alpha_out, (w_out[:, 0]**2 + w_out[:, 1]**2) * tanh_z_out_laplace)

        z_inner = w_inner[:, 0] * x + w_inner[:, 1] * y + b_inner
        tanh_z_inner = np.tanh(z_inner)
        tanh_z_inner_deriv = 1 - tanh_z_inner**2
        tanh_z_inner_laplace = -2 * tanh_z_inner * tanh_z_inner_deriv
        u_inner = np.dot(alpha_inner, tanh_z_inner)
        grad_u_inner_x = np.dot(alpha_inner, w_inner[:, 0] * tanh_z_inner_deriv)
        grad_u_inner_y = np.dot(alpha_inner, w_inner[:, 1] * tanh_z_inner_deriv)
        laplace_u_inner = np.dot(alpha_inner, (w_inner[:, 0]**2 + w_inner[:, 1]**2) * tanh_z_inner_laplace)

        # ================= 摄动解计算 =================
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

        # ================= 雅可比矩阵计算 =================
        if j < M_out + M_inner:  # 区域内部
            if j < M_out:  # 外部区域
                beta_out = x ** 2 + y ** 2 + np.exp(0.5 * u_out)
                beta_x = 2 * x
                beta_y = 2 * y
                beta_out_u = 0.5 * np.exp(0.5 * u_out)
                beta_out_uu = 0.25 * np.exp(0.5 * u_out)
                beta_out_uuu = 0.125 * np.exp(0.5 * u_out)

                f_term1 = (k_val * np.pi) ** 2 * np.sin(k_val * np.pi * x) * np.sin(k_val * np.pi * y)
                f_term2 = (k_val * np.pi) ** 2 / 4 * (
                            np.cos(k_val * np.pi * x) ** 2 * np.sin(k_val * np.pi * y) ** 2 + np.sin(
                        k_val * np.pi * x) ** 2 * np.cos(k_val * np.pi * y) ** 2)
                grad_u_x = 0.5 * k_val * np.pi * np.cos(k_val * np.pi * x) * np.sin(k_val * np.pi * y)
                grad_u_y = 0.5 * k_val * np.pi * np.sin(k_val * np.pi * x) * np.cos(k_val * np.pi * y)
                f_out = (x ** 2 + y ** 2) * f_term1 + np.exp(0.5 * u_out_exact(x, y)) * f_term1 - 2 * (
                        x * grad_u_x + y * grad_u_y) - 0.5 * np.exp(0.5 * u_out_exact(x, y)) * f_term2

                f_out_u = 0.0
                f_out_uu = 0.0

                R_out = (beta_out_u * (grad_u_out_x**2 + grad_u_out_y**2) + beta_out * laplace_u_out + (beta_x * grad_u_out_x + beta_y * grad_u_out_y) + f_out)/epsilon + \
                       2 * beta_out_u * (grad_u_out_x * grad_u_out_px + grad_u_out_y * grad_u_out_py) + \
                       beta_out_uu * (grad_u_out_x**2 + grad_u_out_y**2) * u_out_p + beta_out_u * u_out_p * laplace_u_out + \
                       beta_out * laplace_u_out_p + (beta_x * grad_u_out_px + beta_y * grad_u_out_py) + u_out_p * f_out_u + \
                       epsilon * (2 * u_out_p * beta_out_uu * (grad_u_out_x * grad_u_out_px + grad_u_out_y * grad_u_out_py) + \
                                  0.5 * u_out_p**2 * beta_out_uuu * (grad_u_out_x**2 + grad_u_out_y**2) + 0.5 * u_out_p**2 * beta_out_uu * laplace_u_out + \
                                  beta_out_u * (grad_u_out_px**2 + grad_u_out_py**2) + u_out_p * beta_out_u * laplace_u_out_p + 0.5 * u_out_p**2 * f_out_uu)

                R[j] = R_out

                J_out = 2 * beta_out_u * (grad_u_out_x * w_out1[:, 0] * cos_z_out_p + grad_u_out_y * w_out1[:, 1] * cos_z_out_p) +\
                                   beta_out_uu * (grad_u_out_x**2 + grad_u_out_y**2) * sin_z_out_p + beta_out_u * laplace_u_out * sin_z_out_p + \
                                   beta_out * (w_out1[:, 0]**2 + w_out1[:, 1]**2) * (-sin_z_out_p) + f_out_u * sin_z_out_p + \
                                   (beta_x * w_out1[:, 0] * cos_z_out_p + beta_y * w_out1[:, 1] * cos_z_out_p) + \
                                   epsilon * (2 * sin_z_out_p * beta_out_uu * (grad_u_out_x * grad_u_out_px + grad_u_out_y * grad_u_out_py) + \
                                              2 * u_out_p * beta_out_uu * (grad_u_out_x * w_out1[:, 0] + grad_u_out_y * w_out1[:, 1]) * cos_z_out_p + \
                                              sin_z_out_p * u_out_p * beta_out_uuu * (grad_u_out_x**2 + grad_u_out_y**2) + sin_z_out_p * u_out_p * beta_out_uu * laplace_u_out + \
                                              2 * beta_out_u * (grad_u_out_px * w_out1[:, 0] * cos_z_out_p + grad_u_out_py * w_out1[:, 1] * cos_z_out_p) + \
                                              beta_out_u * laplace_u_out_p * sin_z_out_p + beta_out_u * u_out_p * (w_out1[:, 0]**2 + w_out1[:, 1]**2) * -sin_z_out_p + \
                                              u_out_p * sin_z_out_p * f_out_uu)

                J[j, :N_perturb] = J_out

            else:  # 内部区域
                f_inner = 4 * (np.sin(u_inner) + 1 - (x ** 2 + y ** 2) * np.cos(u_inner))
                f_inner_u = 4 * (np.cos(u_inner) + (x ** 2 + y ** 2) * np.sin(u_inner))
                f_inner_uu = 4 * (-np.sin(u_inner) + (x**2 + y**2) * np.cos(u_inner))

                beta_inner = 1 + np.sin(u_inner)
                beta_inner_u = np.cos(u_inner)
                beta_inner_uu = -np.sin(u_inner)
                beta_inner_uuu = -np.cos(u_inner)

                R_inner = (beta_inner_u * (grad_u_inner_x ** 2 + grad_u_inner_y ** 2) + beta_inner * laplace_u_inner + f_inner) / epsilon + \
                       2 * beta_inner_u * (grad_u_inner_x * grad_u_inner_px + grad_u_inner_y * grad_u_inner_py) + \
                       beta_inner_uu * (grad_u_inner_x ** 2 + grad_u_inner_y ** 2) * u_inner_p + beta_inner_u * u_inner_p * laplace_u_inner + \
                       beta_inner * laplace_u_inner_p + u_inner_p * f_inner_u + \
                       epsilon * (2 * u_inner_p * beta_inner_uu * (grad_u_inner_x * grad_u_inner_px + grad_u_inner_y * grad_u_inner_py) + \
                                  0.5 * u_inner_p ** 2 * beta_inner_uuu * (grad_u_inner_x ** 2 + grad_u_inner_y ** 2) + 0.5 * u_inner_p ** 2 * beta_inner_uu * laplace_u_inner + \
                                  beta_inner_u * (grad_u_inner_px ** 2 + grad_u_inner_py ** 2) + u_inner_p * beta_inner_u * laplace_u_inner_p + 0.5 * u_inner_p ** 2 * f_inner_uu)

                R[j] = R_inner

                J_inner = 2 * beta_inner_u * (grad_u_inner_x * w_inner1[:, 0] * cos_z_inner_p + grad_u_inner_y * w_inner1[:, 1] * cos_z_inner_p) + \
                                   beta_inner_uu * (grad_u_inner_x ** 2 + grad_u_inner_y ** 2) * sin_z_inner_p + beta_inner_u * laplace_u_inner * sin_z_inner_p + \
                                   beta_inner * (w_inner1[:, 0] ** 2 + w_inner1[:, 1] ** 2) * (-sin_z_inner_p) + f_inner_u * sin_z_inner_p + \
                                   epsilon * (2 * sin_z_inner_p * beta_inner_uu * (grad_u_inner_x * grad_u_inner_px + grad_u_inner_y * grad_u_inner_py) + \
                                              2 * u_inner_p * beta_inner_uu * (grad_u_inner_x * w_inner1[:, 0] + grad_u_inner_y * w_inner1[:, 1]) * cos_z_inner_p + \
                                              sin_z_inner_p * u_inner_p * beta_inner_uuu * (grad_u_inner_x ** 2 + grad_u_inner_y ** 2) + sin_z_inner_p * u_inner_p * beta_inner_uu * laplace_u_inner + \
                                              2 * beta_inner_u * (grad_u_inner_px * w_inner1[:, 0] * cos_z_inner_p + grad_u_inner_py * w_inner1[:, 1] * cos_z_inner_p) + \
                                              beta_inner_u * laplace_u_inner_p * sin_z_inner_p + beta_inner_u * u_inner_p * (w_inner1[:, 0] ** 2 + w_inner1[:, 1] ** 2) * -sin_z_inner_p + \
                                              u_inner_p * sin_z_inner_p * f_inner_uu)

                J[j, N_perturb:] = J_inner

        else:  # 界面和边界条件
            gamma_idx = j - (M_out + M_inner)
            if gamma_idx < 2 * M_gamma:  # 界面条件
                # 梅花界面法向
                nx, ny = normal_vec(x, y)

                if gamma_idx < M_gamma:  # 连续性条件
                    R_gamman = ((u_out - u_inner - (u_out_exact(x, y) - u_inner_exact(x, y)))/epsilon + (u_out_p - u_inner_p))
                    R[j] = R_gamman * gamma

                    J[j, :N_perturb] = sin_z_out_p * gamma
                    J[j, N_perturb:] = -sin_z_inner_p * gamma

                else:  # 通量条件
                    beta_out = x ** 2 + y ** 2 + np.exp(0.5 * u_out)
                    beta_out_u = 0.5 * np.exp(0.5 * u_out)
                    beta_out_uu = 0.25 * np.exp(0.5 * u_out)

                    beta_inner = 1 + np.sin(u_inner)
                    beta_inner_u = np.cos(u_inner)
                    beta_inner_uu = -np.sin(u_inner)

                    gout_n = grad_u_out_x * nx + grad_u_out_y * ny
                    ginn_n = grad_u_inner_x * nx + grad_u_inner_y * ny
                    gout_np = grad_u_out_px * nx + grad_u_out_py * ny
                    ginn_np = grad_u_inner_px * nx + grad_u_inner_py * ny

                    R_gammad = ((beta_out * gout_n - beta_inner * ginn_n - v(x, y))/epsilon + \
                           (beta_out_u * u_out_p * gout_n + beta_out * gout_np - \
                            beta_inner_u * u_inner_p * ginn_n - beta_inner * ginn_np) + epsilon * \
                            (0.5 * u_out_p**2 * beta_out_uu * gout_n + u_out_p * beta_out_u * gout_np - \
                             0.5 * u_inner_p**2 * beta_inner_uu * ginn_n - u_inner_p * beta_inner_u * ginn_np))

                    R[j] = R_gammad * gamma

                    J_gammad_out = ((beta_out_u * sin_z_out_p * gout_n + beta_out * (w_out1[:, 0] * nx + w_out1[:, 1] * ny) * cos_z_out_p) + epsilon * \
                                        (u_out_p * sin_z_out_p * beta_out_uu * gout_n + sin_z_out_p * beta_out_u * gout_np + \
                                         u_out_p * beta_out_u * (w_out1[:, 0] * nx + w_out1[:, 1] * ny) * cos_z_out_p))

                    J[j, :N_perturb] = J_gammad_out * gamma

                    J_gammad_inner = -((beta_inner_u * sin_z_inner_p * ginn_n + beta_inner * (w_inner1[:, 0] * nx + w_inner1[:, 1] * ny) * cos_z_inner_p) + epsilon * \
                                        (u_inner_p * sin_z_inner_p * beta_inner_uu * ginn_n + sin_z_inner_p * beta_inner_u * ginn_np + \
                                         u_inner_p * beta_inner_u * (w_inner1[:, 0] * nx + w_inner1[:, 1] * ny) * cos_z_inner_p))

                    J[j, N_perturb:] = J_gammad_inner * gamma
            else:  # 边界条件
                R_bc1 = (u_out + u_out_p * epsilon - u_out_exact(x, y)) / epsilon
                R[j] = R_bc1 * gamma
                J[j, :N_perturb] = sin_z_out_p * gamma

    return R, J

# ------------------------- 测试网格（用于误差评估） -------------------------
N_test = 501
x_test = np.linspace(-1, 1, N_test)
y_test = np.linspace(-1, 1, N_test)
xx, yy = np.meshgrid(x_test, y_test)
points = np.column_stack((xx.ravel(), yy.ravel()))

# 区域划分：基于梅花界面的φ
rr_test = np.hypot(points[:, 0], points[:, 1])
theta_test = np.arctan2(points[:, 1], points[:, 0])
R_test = R_theta(theta_test)
is_external = rr_test >= R_test
external_points_test = points[is_external]
internal_points_test = points[~is_external]

# 精确解
u_exact = np.where(phi_levelset(xx, yy) >= 0, u_out_exact(xx, yy), u_inner_exact(xx, yy))
u_exact_norm = np.linalg.norm(u_exact.ravel())

# ------------------------- 预测函数（ELM输出） -------------------------
def predict(points, w, b, alpha):
    z = np.dot(points, w.T) + b
    return np.dot(np.tanh(z), alpha)

def predict1(points, w, b, alpha):
    z = np.dot(points, w.T) + b
    return np.dot(np.sin(z), alpha)

# ------------------------- 主问题牛顿迭代 -------------------------
k_val = 1
alpha_out = np.zeros(N)
alpha_inner = np.zeros(N)
tol = 1e-10
max_iter = 10
external_points, internal_points, interface_points, boundary_points = generate_points(51, 51*2, 51*4)
main_residual_history = []
main_l2_history = []
delta_threshold = 1e-4
prev_residual = float('inf')

for k in range(max_iter):
    R, J = compute_residual_and_jacobian(alpha_out, alpha_inner, external_points, internal_points, interface_points, boundary_points)
    M_total = len(R)
    residual_norm = np.linalg.norm(R) / np.sqrt(M_total)
    main_residual_history.append(residual_norm)
    print(f"Iter {k}: Residual norm (RMSE) = {residual_norm:.4e}")

    # 主阶段当前相对L2误差（全域）
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

# 主阶段预测（用于二阶段摄动）
u_pred_main = np.zeros(len(points))
u_pred_main[is_external] = predict(external_points_test, w_out, b_out, alpha_out)
u_pred_main[~is_external] = predict(internal_points_test, w_inner, b_inner, alpha_inner)
u_pred_main = u_pred_main.reshape(xx.shape)

# ------------------------- 摄动阶段（牛顿 + ELM） -------------------------
perturb_residual_history = []
perturb_l2_history = []
epsilon = 1e-4
alpha_out_p = np.zeros(N_perturb)
alpha_inner_p = np.zeros(N_perturb)
max_perturb_iter = 5
perturb_tol = 1e-10

# 二阶段使用更稠密的点
external_points_p, internal_points_p, interface_points_p, boundary_points_p = generate_points(101, 101*2, 101*4)

prev_perturb_residual = float('inf')
start_time = time.time()

for k in range(max_perturb_iter):
    R_p, J_p = compute_perturb_residual(alpha_out_p, alpha_inner_p, alpha_out, alpha_inner, external_points_p, internal_points_p, interface_points_p, boundary_points_p)
    M_total_p = len(R_p)
    res_norm = np.linalg.norm(R_p) / np.sqrt(M_total_p)
    print(f"Perturb Iter {k}: Residual (RMSE) = {res_norm:.3e}")
    perturb_residual_history.append(res_norm)

    # 当前总相对L2误差（主解 + ε*摄动）
    u_pred1_current = np.zeros(len(points))
    u_pred1_current[is_external] = predict1(external_points_test, w_out1, b_out1, alpha_out_p)
    u_pred1_current[~is_external] = predict1(internal_points_test, w_inner1, b_inner1, alpha_inner_p)
    u_pred1_current = u_pred1_current.reshape(xx.shape)
    u_total = u_pred_main + epsilon * u_pred1_current
    error_total = u_exact - u_total
    rel_l2_total = np.linalg.norm(error_total.ravel()) / u_exact_norm
    perturb_l2_history.append(rel_l2_total)

    residual_diff = abs(res_norm - prev_perturb_residual) / abs(res_norm) if res_norm != 0 else float('inf')
    if residual_diff < delta_threshold:
        print(f"Perturb iteration stopped: residual difference {residual_diff:.4e} < {delta_threshold}")
        break
    prev_perturb_residual = res_norm

    if res_norm < perturb_tol:
        break

    delta_p, _, _, _ = lstsq(J_p, -R_p, cond=1e-14)
    alpha_out_p += delta_p[:N_perturb]
    alpha_inner_p += delta_p[N_perturb:]

end_time = time.time()
elapsed = end_time - start_time
print(f"Elapsed time: {elapsed:.6f} seconds")

# ------------------------- 最终预测与误差分析 -------------------------
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