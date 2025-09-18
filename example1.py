import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import time
def generate_points(N_sample, N_interface, Mb):
    x = np.linspace(-1, 1, 2*N_sample)
    y = np.linspace(0, 1, N_sample)
    xx, yy = np.meshgrid(x, y)
    index = np.where(xx >= 0)
    external_x = xx[index]
    external_y = yy[index]
    external_points = np.column_stack((external_x, external_y))

    index = np.where(xx <= 0)
    internal_x = xx[index]
    internal_y = yy[index]
    internal_points = np.column_stack((internal_x, internal_y))

    interface_points = np.column_stack((np.zeros(N_interface), np.linspace(0, 1, N_interface)))

    m = Mb // 8  # 每边点数
    left = np.column_stack([-np.ones(2*m), np.linspace(0, 1, 2*m)])
    right = np.column_stack([np.ones(2*m), np.linspace(0, 1, 2*m)])
    bottom1 = np.column_stack([np.linspace(-1.0, 0.0, 2*m), np.zeros(2*m)])
    top1 = np.column_stack([np.linspace(-1.0, 0.0, 2*m), np.ones(2*m)])
    bottom2 = np.column_stack([np.linspace(0.0, 1, 2*m), np.zeros(2*m)])
    top2 = np.column_stack([np.linspace(0.0, 1, 2*m), np.ones(2*m)])
    boundary_points1 = np.concatenate([left, bottom1, top1], axis=0)
    boundary_points2 = np.concatenate([right, bottom2, top2], axis=0)

    return external_points, internal_points, interface_points, boundary_points1, boundary_points2
def u_out_exact(x, y):
    return np.exp(-x) * x**2 * y**2
def u_inner_exact(x, y):
    return x**2 * y**2
# 初始化ELM参数
np.random.seed(42)
N = 100
N_perturb = 400
gamma_bc = 500
gamma_interface = 1000
w_out = 1.0 * np.random.randn(N, 2)  # 隐层权重
b_out = 0.1 * np.random.randn(N)           # 隐层偏置
w_inner = 1.0 * np.random.randn(N, 2)
b_inner = 0.1 * np.random.randn(N)

w_out1 = 5*np.pi * np.random.randn(N_perturb, 2)  # 隐层权重
w_inner1 = 5*np.pi * np.random.randn(N_perturb, 2)  # 隐层权重
b_out1 = 1.0 * np.random.randn(N_perturb)
b_inner1 = 1.0 * np.random.randn(N_perturb)           # 隐层偏置
gamma = 100

def compute_residual_and_jacobian(alpha_out, alpha_inner, external_points, internal_points, interface_points, boundary_points1, boundary_points2):
    M_out = len(external_points)
    M_inner = len(internal_points)
    M_gamma = len(interface_points)
    M_boundary1 = len(boundary_points1)
    M_boundary2 = len(boundary_points2)

    M_total = M_out + M_inner + M_gamma*2 +  M_boundary1 + M_boundary2
    R = np.zeros(M_total)
    J = np.zeros((M_total, 2 * N))

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

        f_out = -2 * np.exp(-x) * (x**2 + y**2) + 4 * np.exp(-x) * x * y**2 - np.exp(-x) * x**2 * y**2 - 5 * np.exp(-3 * x) * (x**4*y**4) * (x**2 + y**2) + \
            6 * np.exp(-3*x)*x**5 *y**6 - 1.5 * u_out**3
        f_out_u = -3 * 1.5 * u_out**2

        beta_out = 1 + 0.5 * u_out**2
        beta_out_u = u_out
        beta_out_uu = 1.0
        # 残差计算
        R_term = beta_out * laplace_u_out + beta_out_u * grad_out_squared + f_out
        R[j] = R_term
        # 雅可比计算
        term1_part1 = beta_out_u * laplace_u_out * tanh_z_out
        term1_part2 = beta_out * (w_out[:, 0] ** 2 + w_out[:, 1] ** 2) * tanh_z_out_laplace
        term2_part1 = beta_out_uu * grad_out_squared * tanh_z_out
        term2_part2 = 2 * beta_out_u * (
                    grad_u_out_x * w_out[:, 0] * tanh_z_out_deriv + grad_u_out_y * w_out[:, 1] * tanh_z_out_deriv)

        J[j, :N] = (term1_part1 + term1_part2 + term2_part1 + term2_part2 + f_out_u * tanh_z_out)

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

        f_inner = -2 * (1 + 3 * u_inner) * (x**2 + y**2)
        f_inner_u = -6 * (x**2 + y**2)

        beta_inner = 1 + u_inner
        beta_inner_u = 1.0
        beta_inner_uu = 0.0
        # 残差计算
        R_term = beta_inner * laplace_u_inner + beta_inner_u * grad_inner_squared + f_inner
        R[j + M_out] = R_term
        # 雅可比计算
        term1_part1 = beta_inner_u * laplace_u_inner * tanh_z_inner
        term1_part2 = beta_inner * (w_inner[:, 0] ** 2 + w_inner[:, 1] ** 2) * tanh_z_inner_laplace
        term2_part1 = beta_inner_uu * grad_inner_squared * tanh_z_inner
        term2_part2 = 2 * beta_inner_u * (grad_u_inner_x * w_inner[:, 0] * tanh_z_inner_deriv + grad_u_inner_y * w_inner[:, 1] * tanh_z_inner_deriv)

        J[j + M_out, N:] = (term1_part1 + term1_part2 + term2_part1 + term2_part2 + f_inner_u * tanh_z_inner)

    for j in range(M_gamma):
        x, y = interface_points[j]
        z_out = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        tanh_z_out = np.tanh(z_out)
        tanh_z_out_deriv = 1 - tanh_z_out ** 2
        u_out = np.dot(alpha_out, tanh_z_out)
        grad_u_out_x = np.dot(alpha_out, w_out[:, 0] * tanh_z_out_deriv)
        grad_u_out_y = np.dot(alpha_out, w_out[:, 1] * tanh_z_out_deriv)
        beta_out = 1 + 0.5 * u_out**2
        beta_out_u = u_out

        z_inner = w_inner[:, 0] * x + w_inner[:, 1] * y + b_inner
        tanh_z_inner = np.tanh(z_inner)
        tanh_z_inner_deriv = 1 - tanh_z_inner ** 2
        u_inner = np.dot(alpha_inner, tanh_z_inner)
        grad_u_inner_x = np.dot(alpha_inner, w_inner[:, 0] * tanh_z_inner_deriv)
        grad_u_inner_y = np.dot(alpha_inner, w_inner[:, 1] * tanh_z_inner_deriv)
        beta_inner = 1 + u_inner
        beta_inner_u = 1.0

        R_gamman = (u_out - u_inner - (u_out_exact(x, y) - u_inner_exact(x, y)))
        R[j + M_out + M_inner] = R_gamman * gamma_interface
        J[j + M_out + M_inner, :N] = tanh_z_out * gamma_interface
        J[j + M_out + M_inner, N:] = -tanh_z_inner * gamma_interface

        R_gammad = (beta_out * grad_u_out_x - beta_inner * grad_u_inner_x)
        R[j + M_out + M_inner + M_gamma] = R_gammad * gamma_interface
        term_out = (beta_out_u * grad_u_out_x * tanh_z_out) + (beta_out * w_out[:, 0] * tanh_z_out_deriv)
        J[j + M_out + M_inner + M_gamma, :N] = term_out * gamma_interface

        # 对alpha_inner的导数
        term_inner = (beta_inner_u * grad_u_inner_x * tanh_z_inner) + (beta_inner * w_inner[:, 0] * tanh_z_inner_deriv)
        J[j + M_out + M_inner + M_gamma, N:] = -term_inner * gamma_interface

    for j in range(M_boundary2):
        x, y = boundary_points2[j]
        z_out = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        tanh_z_out = np.tanh(z_out)
        u_out = np.dot(alpha_out, tanh_z_out)
        R_bc1 = (u_out - u_out_exact(x, y))

        R[j + M_out + M_inner + M_gamma * 2] = R_bc1 * gamma_bc
        J[j + M_out + M_inner + M_gamma * 2, :N] = tanh_z_out * gamma_bc

    for j in range(M_boundary1):
        x, y = boundary_points1[j]
        z_inner = w_inner[:, 0] * x + w_inner[:, 1] * y + b_inner
        tanh_z_inner = np.tanh(z_inner)
        u_inner = np.dot(alpha_inner, tanh_z_inner)
        R_bc2 = (u_inner - u_inner_exact(x, y))

        R[j + M_out + M_inner + M_gamma * 2 + M_boundary1] = R_bc2 * gamma_bc
        J[j + M_out + M_inner + M_gamma * 2 + M_boundary1, N:] = tanh_z_inner * gamma_bc
    return R, J

def compute_perturb_residual(alpha_out_p, alpha_inner_p, alpha_out, alpha_inner, X_out, X_inner, X_gamma, X_omega1, X_omega2):

    X = np.vstack([X_out, X_inner, X_gamma, X_gamma, X_omega1, X_omega2])

    M_out, M_inner = len(X_out), len(X_inner)
    M_gamma = len(X_gamma)
    M_omega1 = len(X_omega1)
    M_omega2 = len(X_omega2)
    M_total = M_out + M_inner + 2 * M_gamma + M_omega1 + M_omega2

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
                # 残差项
                beta_out = 1 + 0.5 * u_out**2
                beta_out_u = u_out
                beta_out_uu = 1.0
                beta_out_uuu = 0
                f_out = -2 * np.exp(-x) * (x**2 + y**2) + 4 * np.exp(-x) * x * y**2 - np.exp(-x) * x**2 * y**2 - 5 * np.exp(-3 * x) * (x**4*y**4) * (x**2 + y**2) + \
            6 * np.exp(-3*x)*x**5 *y**6 - 1.5 * u_out**3
                f_out_u = -4.5 * u_out**2
                f_out_uu = -9 * u_out
                R_out = (beta_out_u * (grad_u_out_x**2 + grad_u_out_y**2) + beta_out * laplace_u_out + f_out)/epsilon + \
                       2 * beta_out_u * (grad_u_out_x * grad_u_out_px + grad_u_out_y * grad_u_out_py) + \
                       beta_out_uu * (grad_u_out_x**2 + grad_u_out_y**2) * u_out_p + beta_out_u * u_out_p * laplace_u_out + \
                       beta_out * laplace_u_out_p + u_out_p * f_out_u + \
                       epsilon * (2 * u_out_p * beta_out_uu * (grad_u_out_x * grad_u_out_px + grad_u_out_y * grad_u_out_py) + \
                                  0.5 * u_out_p**2 * beta_out_uuu * (grad_u_out_x**2 + grad_u_out_y**2) + 0.5 * u_out_p**2 * beta_out_uu * laplace_u_out + \
                                  beta_out_u * (grad_u_out_px**2 + grad_u_out_py**2) + u_out_p * beta_out_u * laplace_u_out_p + 0.5 * u_out_p**2 * f_out_uu)

                R[j] = R_out

                # 雅可比矩阵（仅对alpha_out_p求导）
                J_out = 2 * beta_out_u * (grad_u_out_x * w_out1[:, 0] * cos_z_out_p + grad_u_out_y * w_out1[:, 1] * cos_z_out_p) +\
                                   beta_out_uu * (grad_u_out_x**2 + grad_u_out_y**2) * sin_z_out_p + beta_out_u * laplace_u_out * sin_z_out_p + \
                                   beta_out * (w_out1[:, 0]**2 + w_out1[:, 1]**2) * (-sin_z_out_p) + f_out_u * sin_z_out_p + \
                                   epsilon * (2 * sin_z_out_p * beta_out_uu * (grad_u_out_x * grad_u_out_px + grad_u_out_y * grad_u_out_py) + \
                                              2 * u_out_p * beta_out_uu * (grad_u_out_x * w_out1[:, 0] + grad_u_out_y * w_out1[:, 1]) * cos_z_out_p + \
                                              sin_z_out_p * u_out_p * beta_out_uuu * (grad_u_out_x**2 + grad_u_out_y**2) + sin_z_out_p * u_out_p * beta_out_uu * laplace_u_out + \
                                              2 * beta_out_u * (grad_u_out_px * w_out1[:, 0] * cos_z_out_p + grad_u_out_py * w_out1[:, 1] * cos_z_out_p) + \
                                              beta_out_u * laplace_u_out_p * sin_z_out_p + beta_out_u * u_out_p * (w_out1[:, 0]**2 + w_out1[:, 1]**2) * -sin_z_out_p + \
                                              u_out_p * sin_z_out_p * f_out_uu)

                J[j, :N_perturb] = J_out

            else:  # 内部区域
                f_inner = -2 * (1 + 3 * u_inner) * (x**2 + y**2)
                f_inner_u = -6 * (x**2 + y**2)
                f_inner_uu = 0.0

                beta_inner = 1 + u_inner
                beta_inner_u = 1.0
                beta_inner_uu = 0.0
                beta_inner_uuu = 0.0

                R_inner = (beta_inner_u * (grad_u_inner_x ** 2 + grad_u_inner_y ** 2) + beta_inner * laplace_u_inner + f_inner) / epsilon + \
                       2 * beta_inner_u * (grad_u_inner_x * grad_u_inner_px + grad_u_inner_y * grad_u_inner_py) + \
                       beta_inner_uu * (grad_u_inner_x ** 2 + grad_u_inner_y ** 2) * u_inner_p + beta_inner_u * u_inner_p * laplace_u_inner + \
                       beta_inner * laplace_u_inner_p + u_inner_p * f_inner_u + \
                       epsilon * (2 * u_inner_p * beta_inner_uu * (grad_u_inner_x * grad_u_inner_px + grad_u_inner_y * grad_u_inner_py) + \
                                  0.5 * u_inner_p ** 2 * beta_inner_uuu * (grad_u_inner_x ** 2 + grad_u_inner_y ** 2) + 0.5 * u_inner_p ** 2 * beta_inner_uu * laplace_u_inner + \
                                  beta_inner_u * (grad_u_inner_px ** 2 + grad_u_inner_py ** 2) + u_inner_p * beta_inner_u * laplace_u_inner_p + 0.5 * u_inner_p ** 2 * f_inner_uu)

                R[j] = R_inner

                # 雅可比矩阵（仅对alpha_inner_p求导）
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
                if gamma_idx < M_gamma:  # 连续性条件
                    R_gamman = ((u_out - u_inner - (u_out_exact(x, y) - u_inner_exact(x, y)))/epsilon + (u_out_p - u_inner_p))

                    R[j] = R_gamman * gamma

                    # 雅可比矩阵
                    J[j, :N_perturb] = sin_z_out_p * gamma
                    J[j, N_perturb:] = -sin_z_inner_p * gamma

                else:  # 通量条件
                    # 主通量项
                    beta_out = 1 + 0.5 * u_out**2
                    beta_out_u = u_out
                    beta_out_uu = 1.0
                    beta_inner = 1 + u_inner
                    beta_inner_u = 1.0
                    beta_inner_uu = 0.0

                    R_gammad = ((beta_out * grad_u_out_x - beta_inner * grad_u_inner_x)/epsilon + \
                           (beta_out_u * u_out_p * grad_u_out_x + beta_out * grad_u_out_px - \
                            beta_inner_u * u_inner_p * grad_u_inner_x - beta_inner * grad_u_inner_px) + epsilon * \
                            (0.5 * u_out_p**2 * beta_out_uu * grad_u_out_x + u_out_p * beta_out_u * grad_u_out_px - \
                             0.5 * u_inner_p**2 * beta_inner_uu * grad_u_inner_x - u_inner_p * beta_inner_u * grad_u_inner_px))

                    R[j] = R_gammad * gamma

                    # 雅可比矩阵
                    J_gammad_out = ((beta_out_u * sin_z_out_p * grad_u_out_x + beta_out * w_out1[:, 0] * cos_z_out_p) + epsilon * \
                                        (u_out_p * sin_z_out_p * beta_out_uu * grad_u_out_x + sin_z_out_p * beta_out_u * grad_u_out_px + \
                                         u_out_p * beta_out_u * w_out1[:, 0] * cos_z_out_p))

                    J[j, :N_perturb] = J_gammad_out * gamma
                    J_gammad_inner = -((beta_inner_u * sin_z_inner_p * grad_u_inner_x + beta_inner * w_inner1[:, 0] * cos_z_inner_p) + epsilon * \
                                        (u_inner_p * sin_z_inner_p * beta_inner_uu * grad_u_inner_x + sin_z_inner_p * beta_inner_u * grad_u_inner_px + \
                                         u_inner_p * beta_inner_u * w_inner1[:, 0] * cos_z_inner_p))

                    J[j, N_perturb:] = J_gammad_inner * gamma
            else:  # 边界条件
                bc_id = j - (M_out + M_inner + 2 * M_gamma)
                if bc_id < M_omega1:
                    R_bc2 = (u_inner + u_inner_p * epsilon - u_inner_exact(x, y)) / epsilon
                    R[j] = R_bc2 * gamma
                    J[j, N_perturb:] = sin_z_inner_p * gamma
                else:
                    R_bc1 = (u_out + u_out_p * epsilon - u_out_exact(x, y)) / epsilon
                    R[j] = R_bc1 * gamma
                    J[j, :N_perturb] = sin_z_out_p * gamma

    return R, J


alpha_out = np.zeros(N)  # 初始猜测
alpha_inner = np.zeros(N)
tol = 1e-10
max_iter = 10
external_points, internal_points, interface_points, boundary_points1, boundary_points2 = generate_points(101, 101, 808)
main_residual_history = []
delta_threshold = 1e-6  # 新增: 迭代结束条件阈值 δ
prev_residual = float('inf')
start_time = time.time()
for k in range(max_iter):
    R, J = compute_residual_and_jacobian(alpha_out, alpha_inner, external_points, internal_points, interface_points,
                                         boundary_points1, boundary_points2)
    M_total = len(R)
    residual_norm = np.linalg.norm(R)/ np.sqrt(M_total)
    main_residual_history.append(residual_norm)
    print(f"Iter {k}: Residual norm = {residual_norm:.4e}")

    # 新增: 检查残差变化
    residual_diff = abs(residual_norm - prev_residual)
    if residual_diff < delta_threshold:
        print(f"Main iteration stopped: residual difference {residual_diff:.4e} < {delta_threshold}")
        break
    prev_residual = residual_norm

    if residual_norm < tol:
        break
    # 求解最小二乘问题 Jδβ = -R
    delta_beta, _, _, _ = lstsq(J, -R, cond=1e-12)
    alpha_out += delta_beta[:N]
    alpha_inner += delta_beta[N:]
print(M_total)
perturb_residual_history = []
epsilon = 1e-4
alpha_out_p = np.zeros(N_perturb)  # 初始猜测
alpha_inner_p = np.zeros(N_perturb)

max_perturb_iter = 10
perturb_tol = 1e-10

# 生成更密集的摄动配置点
external_points, internal_points, interface_points, boundary_points1, boundary_points2 = generate_points(126, 251, 2000)

prev_perturb_residual = float('inf')
for k in range(max_perturb_iter):
    R_p, J_p = compute_perturb_residual(alpha_out_p, alpha_inner_p, alpha_out, alpha_inner, external_points,
                                        internal_points, interface_points, boundary_points1, boundary_points2)
    M_total_p = len(R_p)
    res_norm = np.linalg.norm(R_p) / np.sqrt(M_total_p)
    print(f"Perturb Iter {k}: Residual = {res_norm:.3e}")
    perturb_residual_history.append(res_norm)

    # 新增: 检查残差变化
    residual_diff = abs(res_norm - prev_perturb_residual)
    if residual_diff < delta_threshold:
        print(f"Perturb iteration stopped: residual difference {residual_diff:.4e} < {delta_threshold}")
        break
    prev_perturb_residual = res_norm
    if res_norm > 1:
        delta = 1e-14 * res_norm
    else:
        delta = 0.0

    if res_norm < perturb_tol:
        break

    delta_p, _, _, _ = lstsq(J_p, -R_p, cond=None)
    alpha_out_p += delta_p[:N_perturb]
    alpha_inner_p += delta_p[N_perturb:]

end_time = time.time()  # 记录结束时间戳
elapsed = end_time - start_time  # 计算差值
print(f"耗时: {elapsed:.6f} 秒")
print(M_total_p)
# 生成测试网格
N_test = 201
x_test = np.linspace(-1, 1, 2*N_test)
y_test = np.linspace(0, 1, N_test)
xx, yy = np.meshgrid(x_test, y_test)
points = np.column_stack((xx.ravel(), yy.ravel()))

# 划分区域
is_external = points[:, 0] >= 0.0
external_points = points[is_external]
internal_points = points[~is_external]

# 向量化计算预测解
def predict(points, w, b, alpha):
    z = points @ w.T + b
    return np.tanh(z) @ alpha
def predict1(points, w, b, alpha):
    z = points @ w.T + b
    return np.sin(z) @ alpha

u_pred = np.zeros(len(points))
u_pred[is_external] = predict(external_points, w_out, b_out, alpha_out)
u_pred[~is_external] = predict(internal_points, w_inner, b_inner, alpha_inner)
u_pred = u_pred.reshape(xx.shape)

u_pred1 = np.zeros(len(points))
u_pred1[is_external] = predict1(external_points, w_out1, b_out1, alpha_out_p)
u_pred1[~is_external] = predict1(internal_points, w_inner1, b_inner1, alpha_inner_p)
u_pred1 = u_pred1.reshape(xx.shape)

# 计算精确解
u_exact = np.where(xx >= 0.0, u_out_exact(xx, yy), u_inner_exact(xx, yy))
error = u_exact - u_pred
error1 = error - epsilon * u_pred1
# 误差分析
print(f"最大绝对误差: {np.max(error):.4e}")
print(f"相对L2误差: {np.linalg.norm(error)/np.linalg.norm(u_exact):.4e}")
print(f"Stage2相对L2误差: {np.linalg.norm(error1)/np.linalg.norm(u_exact):.4e}")
print(f"Stage2最大绝对误差: {np.max(error1):.4e}")
