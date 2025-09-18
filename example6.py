import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import time

# 全局参数
k_val = 1  # 精确解参数

# 生成采样点
def generate_points(N_sample, N_interface, Mb):
    # N_sample: 每个方向网格数
    # N_interface: 界面点个数
    # Mb: 边界点总数(将按4边均分)
    x = np.linspace(-1, 1, N_sample)
    y = np.linspace(-1, 1, N_sample)
    xx, yy = np.meshgrid(x, y)
    index = np.where(xx**2 + yy**2 >= 0.25)
    external_x = xx[index]
    external_y = yy[index]
    external_points = np.column_stack((external_x, external_y))

    index = np.where(xx**2 + yy**2 <= 0.25)
    internal_x = xx[index]
    internal_y = yy[index]
    internal_points = np.column_stack((internal_x, internal_y))

    theta = np.linspace(0, 2*np.pi, N_interface)
    interface_points = np.column_stack((0.5*np.cos(theta), 0.5*np.sin(theta)))

    m = Mb // 8  # 每边点数
    left = np.column_stack([-np.ones(m), np.linspace(-1, 1, m)])
    right = np.column_stack([np.ones(m), np.linspace(-1, 1, m)])
    bottom = np.column_stack([np.linspace(-1.0, 1.0, m), -np.ones(m)])
    top = np.column_stack([np.linspace(-1.0, 1.0, m), np.ones(m)])
    boundary_points = np.concatenate([left, right, bottom, top], axis=0)

    return external_points, internal_points, interface_points, boundary_points

# 精确解
def u_out_exact(x, y):
    # 外部区域精确解
    return 0.5 * np.sin(k_val * np.pi * x) * np.sin(k_val * np.pi * y) + 0.25

def u_inner_exact(x, y):
    # 内部区域精确解
    return 0.25 - (x**2 + y**2)

# 外部精确解的一阶二阶导数（梯度与Hessian）
def out_exact_derivatives(x, y):
    # 返回 u, ux, uy, uxx, uyy, uxy
    kpi = k_val * np.pi
    sinx = np.sin(kpi * x)
    cosx = np.cos(kpi * x)
    siny = np.sin(kpi * y)
    cosy = np.cos(kpi * y)
    u = 0.5 * sinx * siny + 0.25
    ux = 0.5 * kpi * cosx * siny
    uy = 0.5 * kpi * sinx * cosy
    uxx = -0.5 * (kpi**2) * sinx * siny
    uyy = -0.5 * (kpi**2) * sinx * siny
    uxy = 0.5 * (kpi**2) * cosx * cosy
    return u, ux, uy, uxx, uyy, uxy

# 内部精确解的导数（梯度与Hessian）
def inner_exact_derivatives(x, y):
    # 返回 u, ux, uy, uxx, uyy, uxy
    u = 0.25 - (x**2 + y**2)
    ux = -2.0 * x
    uy = -2.0 * y
    uxx = -2.0
    uyy = -2.0
    uxy = 0.0
    return u, ux, uy, uxx, uyy, uxy

# 新问题的通量跳跃右端（非单位法向，按算法使用 n'=(x,y)）
def v(x, y):
    # 外部侧：beta_out = 1 + |∇u|^2，通量项：beta_out * (∇u · (x,y))
    uo, uox, uoy, uoxx, uoyy, uoxy = out_exact_derivatives(x, y)
    beta_out = 1.0 + (uox**2 + uoy**2)
    flux_out = beta_out * (uox * x + uoy * y)
    # 内部侧：beta_in = 1 + u^2，通量项：beta_in * (∇u · (x,y))
    ui, uix, uiy, uixx, uiyy, uixy = inner_exact_derivatives(x, y)
    beta_in = 1.0 + ui**2
    flux_in = beta_in * (uix * x + uiy * y)
    # 跳跃右端（外 - 内）
    return flux_out - flux_in

# 外部区域新源项 f_out，使得 u_out_exact 是解
def f_out_rhs(x, y):
    # f_out = - [ (1 + |∇u|^2) Δu + 2 ∇u^T H ∇u ]，均使用精确解的导数
    uo, uox, uoy, uoxx, uoyy, uoxy = out_exact_derivatives(x, y)
    lap = uoxx + uoyy
    g_norm2 = uox**2 + uoy**2
    S = (uox**2) * uoxx + 2.0 * uox * uoy * uoxy + (uoy**2) * uoyy
    return -((1.0 + g_norm2) * lap + 2.0 * S)

# 内部区域新源项 f_inner，使得 u_inner_exact 是解
def f_inner_rhs(x, y):
    # β = 1 + u^2, β_u = 2u, Δu = -4, |∇u|^2 = 4 r^2, u = 0.25 - r^2
    r2 = x**2 + y**2
    u = 0.25 - r2
    beta = 1.0 + u**2
    beta_u = 2.0 * u
    lap = -4.0
    grad2 = 4.0 * r2
    # f = - (β Δu + β_u |∇u|^2)
    return - (beta * lap + beta_u * grad2)

# 初始化ELM参数
np.random.seed(42)
N = 500
N_perturb = 4000
gamma_bc = 1000
gamma_interface = 1000

# 主网络(外/内)参数
w_out = 1.0 * np.random.randn(N, 2)    # 隐层权重
b_out = 0.1 * np.random.randn(N)       # 隐层偏置
w_inner = 1.0 * np.random.randn(N, 2)
b_inner = 0.1 * np.random.randn(N)

# 摄动网络参数
w_out1 = 7*np.pi * np.random.randn(N_perturb, 2)
w_inner1 = 7*np.pi * np.random.randn(N_perturb, 2)
b_out1 = 1.0 * np.random.randn(N_perturb)
b_inner1 = 1.0 * np.random.randn(N_perturb)

gamma = 100  # 摄动阶段罚参数

# 主阶段残差与雅可比（基于新的 β_out=1+|∇u|^2, β_in=1+u^2）
def compute_residual_and_jacobian(alpha_out, alpha_inner, external_points, internal_points, interface_points, boundary_points):
    M_out = len(external_points)
    M_inner = len(internal_points)
    M_gamma = len(interface_points)
    M_boundary = len(boundary_points)

    M_total = M_out + M_inner + 2 * M_gamma + M_boundary
    R = np.zeros(M_total)
    J = np.zeros((M_total, 2 * N))

    # 外部区域(β_out=1+|∇u|^2), PDE: ∇·(β ∇u) + f_out = 0
    # 展开：R = β Δu + 2 ∇u^T H ∇u + f_out
    for j in range(M_out):
        x, y = external_points[j]
        z = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        th = np.tanh(z)
        th_d = 1.0 - th**2
        th_dd = -2.0 * th * th_d  # 二阶导因子φ''

        # 主解项
        u = np.dot(alpha_out, th)
        ux = np.dot(alpha_out, w_out[:, 0] * th_d)
        uy = np.dot(alpha_out, w_out[:, 1] * th_d)
        uxx = np.dot(alpha_out, (w_out[:, 0]**2) * th_dd)
        uyy = np.dot(alpha_out, (w_out[:, 1]**2) * th_dd)
        uxy = np.dot(alpha_out, (w_out[:, 0] * w_out[:, 1]) * th_dd)
        lap = uxx + uyy

        beta = 1.0 + ux**2 + uy**2
        S = ux**2 * uxx + 2.0 * ux * uy * uxy + uy**2 * uyy

        f_out = f_out_rhs(x, y)

        R_term = beta * lap + 2.0 * S + f_out
        R[j] = R_term

        # 雅可比 w.r.t alpha_out
        bxi = w_out[:, 0] * th_d
        byi = w_out[:, 1] * th_d
        hxi = (w_out[:, 0]**2) * th_dd
        hyi = (w_out[:, 1]**2) * th_dd
        hxyi = (w_out[:, 0] * w_out[:, 1]) * th_dd
        Li = hxi + hyi

        term_beta = 2.0 * (ux * bxi + uy * byi) * lap
        term_lap = beta * Li
        term_S = 2.0 * (2.0 * ux * bxi * uxx + ux**2 * hxi +
                        2.0 * (bxi * uy + ux * byi) * uxy + 2.0 * ux * uy * hxyi +
                        2.0 * uy * byi * uyy + uy**2 * hyi)
        J[j, :N] = term_beta + term_lap + term_S

    # 内部区域(β_in=1+u^2), PDE: ∇·(β ∇u) + f_in = 0
    # 展开：R = β Δu + β_u |∇u|^2 + f_in, 其中 β=1+u^2, β_u=2u
    for j in range(M_inner):
        x, y = internal_points[j]
        z = w_inner[:, 0] * x + w_inner[:, 1] * y + b_inner
        th = np.tanh(z)
        th_d = 1.0 - th**2
        th_dd = -2.0 * th * th_d

        u = np.dot(alpha_inner, th)
        ux = np.dot(alpha_inner, w_inner[:, 0] * th_d)
        uy = np.dot(alpha_inner, w_inner[:, 1] * th_d)
        lap = np.dot(alpha_inner, (w_inner[:, 0]**2 + w_inner[:, 1]**2) * th_dd)

        grad2 = ux**2 + uy**2

        beta = 1.0 + u**2
        beta_u = 2.0 * u
        beta_uu = 2.0

        f_in = f_inner_rhs(x, y)

        R_term = beta * lap + beta_u * grad2 + f_in
        R[j + M_out] = R_term

        # 雅可比 w.r.t alpha_inner
        term1_part1 = beta_u * lap * th
        term1_part2 = beta * (w_inner[:, 0]**2 + w_inner[:, 1]**2) * th_dd
        term2_part1 = beta_uu * grad2 * th
        term2_part2 = 2.0 * beta_u * (ux * w_inner[:, 0] * th_d + uy * w_inner[:, 1] * th_d)
        J[j + M_out, N:] = (term1_part1 + term1_part2 + term2_part1 + term2_part2)

    # 界面条件
    for j in range(M_gamma):
        x, y = interface_points[j]

        # 外部侧
        z_out = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        th_out = np.tanh(z_out)
        th_out_d = 1.0 - th_out**2
        u_out = np.dot(alpha_out, th_out)
        ux_out = np.dot(alpha_out, w_out[:, 0] * th_out_d)
        uy_out = np.dot(alpha_out, w_out[:, 1] * th_out_d)
        beta_out = 1.0 + ux_out**2 + uy_out**2

        # 内部侧
        z_in = w_inner[:, 0] * x + w_inner[:, 1] * y + b_inner
        th_in = np.tanh(z_in)
        th_in_d = 1.0 - th_in**2
        u_in = np.dot(alpha_inner, th_in)
        ux_in = np.dot(alpha_inner, w_inner[:, 0] * th_in_d)
        uy_in = np.dot(alpha_inner, w_inner[:, 1] * th_in_d)
        beta_in = 1.0 + u_in**2
        beta_in_u = 2.0 * u_in

        # 连续性
        R_gamman = (u_out - u_in - (u_out_exact(x, y) - u_inner_exact(x, y)))
        R[j + M_out + M_inner] = R_gamman * gamma_interface
        J[j + M_out + M_inner, :N] = th_out * gamma_interface
        J[j + M_out + M_inner, N:] = -th_in * gamma_interface

        # 通量条件（法向取 n'=(x,y)）
        R_gammad = (beta_out * (ux_out * x + uy_out * y) - beta_in * (ux_in * x + uy_in * y)) - v(x, y)
        R[j + M_out + M_inner + M_gamma] = R_gammad * gamma_interface

        # 外部侧雅可比
        bxi_out = w_out[:, 0] * th_out_d
        byi_out = w_out[:, 1] * th_out_d
        g_dot_n = ux_out * x + uy_out * y
        dPhi_out = (2.0 * (ux_out * bxi_out + uy_out * byi_out)) * g_dot_n + beta_out * (bxi_out * x + byi_out * y)
        J[j + M_out + M_inner + M_gamma, :N] = dPhi_out * gamma_interface

        # 内部侧雅可比
        bxi_in = w_inner[:, 0] * th_in_d
        byi_in = w_inner[:, 1] * th_in_d
        g_in_dot_n = ux_in * x + uy_in * y
        dPhi_in = (beta_in_u * th_in) * g_in_dot_n + beta_in * (bxi_in * x + byi_in * y)
        J[j + M_out + M_inner + M_gamma, N:] = -dPhi_in * gamma_interface  # 减去内部通量

    # 外边界Dirichlet
    for j in range(M_boundary):
        x, y = boundary_points[j]
        z_out = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        th_out = np.tanh(z_out)
        u_out = np.dot(alpha_out, th_out)
        R_bc = (u_out - u_out_exact(x, y))
        R[j + M_out + M_inner + 2 * M_gamma] = R_bc * gamma_bc
        J[j + M_out + M_inner + 2 * M_gamma, :N] = th_out * gamma_bc

    return R, J

# 摄动阶段残差与雅可比（新的β与展开）
def compute_perturb_residual(alpha_out_p, alpha_inner_p, alpha_out, alpha_inner, X_out, X_inner, X_gamma, X_omega):
    # 将点集合并
    X = np.vstack([X_out, X_inner, X_gamma, X_gamma, X_omega])

    M_out, M_inner = len(X_out), len(X_inner)
    M_gamma = len(X_gamma)
    M_omega = len(X_omega)
    M_total = M_out + M_inner + 2 * M_gamma + M_omega

    R = np.zeros(M_total)
    J = np.zeros((M_total, 2 * N_perturb))

    for j in range(M_total):
        x, y = X[j]

        # ========== 主解外部 ==========
        z_out = w_out[:, 0] * x + w_out[:, 1] * y + b_out
        th_out = np.tanh(z_out)
        th_out_d = 1.0 - th_out**2
        th_out_dd = -2.0 * th_out * th_out_d

        u_out = np.dot(alpha_out, th_out)
        ux_out = np.dot(alpha_out, w_out[:, 0] * th_out_d)
        uy_out = np.dot(alpha_out, w_out[:, 1] * th_out_d)
        uxx_out = np.dot(alpha_out, (w_out[:, 0]**2) * th_out_dd)
        uyy_out = np.dot(alpha_out, (w_out[:, 1]**2) * th_out_dd)
        uxy_out = np.dot(alpha_out, (w_out[:, 0] * w_out[:, 1]) * th_out_dd)
        lap_out = uxx_out + uyy_out

        beta0_out = 1.0 + ux_out**2 + uy_out**2
        # H_out g_out 向量
        Hg_out_x = uxx_out * ux_out + uxy_out * uy_out
        Hg_out_y = uxy_out * ux_out + uyy_out * uy_out
        # g^T K g 将在摄动部分计算

        # ========== 主解内部 ==========
        z_in = w_inner[:, 0] * x + w_inner[:, 1] * y + b_inner
        th_in = np.tanh(z_in)
        th_in_d = 1.0 - th_in**2
        th_in_dd = -2.0 * th_in * th_in_d

        u_in = np.dot(alpha_inner, th_in)
        ux_in = np.dot(alpha_inner, w_inner[:, 0] * th_in_d)
        uy_in = np.dot(alpha_inner, w_inner[:, 1] * th_in_d)
        lap_in = np.dot(alpha_inner, (w_inner[:, 0]**2 + w_inner[:, 1]**2) * th_in_dd)
        grad2_in = ux_in**2 + uy_in**2
        beta_in = 1.0 + u_in**2
        beta_in_u = 2.0 * u_in
        beta_in_uu = 2.0

        # ========== 摄动外部（sin激活）==========
        z_out_p = w_out1[:, 0] * x + w_out1[:, 1] * y + b_out1
        s_out = np.sin(z_out_p)
        c_out = np.cos(z_out_p)
        u_out_p = np.dot(alpha_out_p, s_out)
        hx_out = np.dot(alpha_out_p, w_out1[:, 0] * c_out)
        hy_out = np.dot(alpha_out_p, w_out1[:, 1] * c_out)
        lap_out_p = np.dot(alpha_out_p, (w_out1[:, 0]**2 + w_out1[:, 1]**2) * (-s_out))
        Kxx_out = np.dot(alpha_out_p, (w_out1[:, 0]**2) * (-s_out))
        Kyy_out = np.dot(alpha_out_p, (w_out1[:, 1]**2) * (-s_out))
        Kxy_out = np.dot(alpha_out_p, (w_out1[:, 0] * w_out1[:, 1]) * (-s_out))

        # ========== 摄动内部（sin激活）==========
        z_in_p = w_inner1[:, 0] * x + w_inner1[:, 1] * y + b_inner1
        s_in = np.sin(z_in_p)
        c_in = np.cos(z_in_p)
        u_in_p = np.dot(alpha_inner_p, s_in)
        hx_in = np.dot(alpha_inner_p, w_inner1[:, 0] * c_in)
        hy_in = np.dot(alpha_inner_p, w_inner1[:, 1] * c_in)
        lap_in_p = np.dot(alpha_inner_p, (w_inner1[:, 0]**2 + w_inner1[:, 1]**2) * (-s_in))

        if j < M_out + M_inner:
            # 区域内部
            if j < M_out:
                # 外部区域：F(u)=βΔu+2 g^T H g + f_out(x)
                f_out = f_out_rhs(x, y)
                g_dot_h = ux_out * hx_out + uy_out * hy_out
                gKg = ux_out**2 * Kxx_out + 2.0 * ux_out * uy_out * Kxy_out + uy_out**2 * Kyy_out
                Kg_x = Kxx_out * ux_out + Kxy_out * uy_out
                Kg_y = Kxy_out * ux_out + Kyy_out * uy_out
                Hh_x = uxx_out * hx_out + uxy_out * hy_out
                Hh_y = uxy_out * hx_out + uyy_out * hy_out
                h_norm2 = hx_out**2 + hy_out**2

                F0 = beta0_out * lap_out + 2.0 * (ux_out**2 * uxx_out + 2.0 * ux_out * uy_out * uxy_out + uy_out**2 * uyy_out) + f_out
                F1 = 2.0 * g_dot_h * lap_out + beta0_out * lap_out_p + 4.0 * (hx_out * Hg_out_x + hy_out * Hg_out_y) + 2.0 * gKg
                F2 = h_norm2 * lap_out + 2.0 * g_dot_h * lap_out_p + 4.0 * (hx_out * Kg_x + hy_out * Kg_y) + 2.0 * (hx_out * Hh_x + hy_out * Hh_y)

                R[j] = F0 / epsilon + F1 + epsilon * F2

                # 雅可比 (w.r.t alpha_out_p)
                bxi_p = w_out1[:, 0] * c_out
                byi_p = w_out1[:, 1] * c_out
                lap_basis_p = -(w_out1[:, 0]**2 + w_out1[:, 1]**2) * s_out
                Kxx_basis = -(w_out1[:, 0]**2) * s_out
                Kyy_basis = -(w_out1[:, 1]**2) * s_out
                Kxy_basis = -(w_out1[:, 0] * w_out1[:, 1]) * s_out

                # F1部分导数
                A = 2.0 * lap_out * (ux_out * bxi_p + uy_out * byi_p)
                B = beta0_out * lap_basis_p
                C = 4.0 * (bxi_p * Hg_out_x + byi_p * Hg_out_y)
                D = 2.0 * (ux_out**2 * Kxx_basis + 2.0 * ux_out * uy_out * Kxy_basis + uy_out**2 * Kyy_basis)

                # F2部分导数
                E = 2.0 * lap_out * (hx_out * bxi_p + hy_out * byi_p)
                F = 2.0 * ((ux_out * bxi_p + uy_out * byi_p) * lap_out_p + g_dot_h * lap_basis_p)
                G = 4.0 * ((bxi_p * Kg_x + byi_p * Kg_y) +
                           (hx_out * (Kxx_basis * ux_out + Kxy_basis * uy_out) +
                            hy_out * (Kxy_basis * ux_out + Kyy_basis * uy_out)))
                H = 4.0 * (bxi_p * Hh_x + byi_p * Hh_y)

                J[j, :N_perturb] = A + B + C + D + epsilon * (E + F + G + H)

            else:
                # 内部区域：β=1+u^2
                # Taylor展开到二阶(与原算法结构一致)，f_in对u无关 => f_u=f_uu=0
                f_in = f_inner_rhs(x, y)
                beta_u = 2.0 * u_in
                beta_uu = 2.0
                beta = 1.0 + u_in**2

                R_inner = (beta_u * (ux_in**2 + uy_in**2) + beta * lap_in + f_in) / epsilon + \
                          2.0 * beta_u * (ux_in * hx_in + uy_in * hy_in) + \
                          beta_uu * (ux_in**2 + uy_in**2) * u_in_p + beta_u * u_in_p * lap_in + \
                          beta * lap_in_p + \
                          epsilon * (2.0 * u_in_p * beta_uu * (ux_in * hx_in + uy_in * hy_in) +
                                     0.5 * u_in_p**2 * beta_uu * lap_in +
                                     beta_u * (hx_in**2 + hy_in**2) + u_in_p * beta_u * lap_in_p)

                R[j] = R_inner

                # 雅可比（w.r.t alpha_inner_p）
                J_inner = 2.0 * beta_u * (ux_in * w_inner1[:, 0] * c_in + uy_in * w_inner1[:, 1] * c_in) + \
                          beta_uu * (ux_in**2 + uy_in**2) * s_in + beta_u * lap_in * s_in + \
                          beta * (w_inner1[:, 0]**2 + w_inner1[:, 1]**2) * (-s_in) + \
                          epsilon * (2.0 * s_in * beta_uu * (ux_in * hx_in + uy_in * hy_in) +
                                     2.0 * u_in_p * beta_uu * (ux_in * w_inner1[:, 0] + uy_in * w_inner1[:, 1]) * c_in +
                                     s_in * u_in_p * beta_uu * lap_in +
                                     2.0 * beta_u * (w_inner1[:, 0] * c_in * hx_in + w_inner1[:, 1] * c_in * hy_in) +
                                     beta_u * lap_in_p * s_in + beta_u * u_in_p * (w_inner1[:, 0]**2 + w_inner1[:, 1]**2) * (-s_in))

                J[j, N_perturb:] = J_inner

        else:
            # 界面条件/边界条件
            gamma_idx = j - (M_out + M_inner)
            if gamma_idx < 2 * M_gamma:
                # 界面
                if gamma_idx < M_gamma:
                    # 连续性
                    R_gamman = ((u_out - u_in - (u_out_exact(x, y) - u_inner_exact(x, y))) / epsilon + (u_out_p - u_in_p))
                    R[j] = R_gamman * gamma
                    J[j, :N_perturb] = np.sin(z_out_p) * gamma
                    J[j, N_perturb:] = -np.sin(z_in_p) * gamma
                else:
                    # 通量条件：外 - 内 - v = 0
                    # 外部展开
                    A0_out = ux_out * x + uy_out * y
                    A1_out = hx_out * x + hy_out * y
                    g_dot_h_out = ux_out * hx_out + uy_out * hy_out
                    first_out = (2.0 * g_dot_h_out) * A0_out + beta0_out * A1_out
                    second_out = (hx_out**2 + hy_out**2) * A0_out + 2.0 * g_dot_h_out * A1_out

                    # 内部展开
                    A0_in = ux_in * x + uy_in * y
                    A1_in = hx_in * x + hy_in * y
                    first_in = (2.0 * u_in * u_in_p) * A0_in + (1.0 + u_in**2) * A1_in
                    second_in = (u_in_p**2) * A0_in + 2.0 * u_in * u_in_p * A1_in

                    R_gammad = ((beta0_out * A0_out - (1.0 + u_in**2) * A0_in - v(x, y)) / epsilon +
                                (first_out - first_in) +
                                epsilon * (second_out - second_in))
                    R[j] = R_gammad * gamma

                    # 雅可比分别对 alpha_out_p 与 alpha_inner_p
                    # 外部
                    d_first_out = 2.0 * A0_out * (ux_out * (w_out1[:, 0] * c_out) + uy_out * (w_out1[:, 1] * c_out)) + \
                                  beta0_out * (w_out1[:, 0] * c_out * x + w_out1[:, 1] * c_out * y)
                    d_second_out = 2.0 * A0_out * (hx_out * (w_out1[:, 0] * c_out) + hy_out * (w_out1[:, 1] * c_out)) + \
                                   2.0 * (ux_out * (w_out1[:, 0] * c_out) + uy_out * (w_out1[:, 1] * c_out)) * A1_out + \
                                   2.0 * g_dot_h_out * (w_out1[:, 0] * c_out * x + w_out1[:, 1] * c_out * y)
                    J_out_gamma = d_first_out + epsilon * d_second_out
                    J[j, :N_perturb] = J_out_gamma * gamma

                    # 内部
                    d_first_in = 2.0 * u_in * (s_in) * A0_in + (1.0 + u_in**2) * (w_inner1[:, 0] * c_in * x + w_inner1[:, 1] * c_in * y)
                    d_second_in = 2.0 * u_in_p * (s_in) * A0_in + \
                                  2.0 * u_in * (s_in) * A1_in + \
                                  2.0 * u_in * u_in_p * (w_inner1[:, 0] * c_in * x + w_inner1[:, 1] * c_in * y)
                    J_in_gamma = d_first_in + epsilon * d_second_in
                    J[j, N_perturb:] = -J_in_gamma * gamma  # 内部贡献为负
            else:
                # 外边界Dirichlet
                R_bc1 = (u_out + epsilon * u_out_p - u_out_exact(x, y)) / epsilon
                R[j] = R_bc1 * gamma
                J[j, :N_perturb] = np.sin(z_out_p) * gamma

    return R, J

# 预测函数（主网络）
def predict(points, w, b, alpha):
    z = np.dot(points, w.T) + b
    return np.dot(np.tanh(z), alpha)

# 预测函数（摄动网络）
def predict1(points, w, b, alpha):
    z = np.dot(points, w.T) + b
    return np.dot(np.sin(z), alpha)

# 测试网格
N_test = 501
x_test = np.linspace(-1, 1, N_test)
y_test = np.linspace(-1, 1, N_test)
xx, yy = np.meshgrid(x_test, y_test)
points = np.column_stack((xx.ravel(), yy.ravel()))

# 区域划分
is_external = points[:, 0]**2 + points[:, 1]**2 >= 0.25
external_points_test = points[is_external]
internal_points_test = points[~is_external]

# 精确解栅格
u_exact = np.where(xx**2 + yy**2 >= 0.25, u_out_exact(xx, yy), u_inner_exact(xx, yy))
u_exact_norm = np.linalg.norm(u_exact.ravel())

# 主阶段Newton迭代
alpha_out = np.zeros(N)
alpha_inner = np.zeros(N)
tol = 1e-10
max_iter = 10
external_points, internal_points, interface_points, boundary_points = generate_points(81, 101, 81*4)
main_residual_history = []
main_l2_history = []
delta_threshold = 1e-4
prev_residual = float('inf')

print("--- Starting Main Stage ---")
for k in range(max_iter):
    R, J = compute_residual_and_jacobian(alpha_out, alpha_inner, external_points, internal_points, interface_points, boundary_points)
    M_total = len(R)
    residual_norm = np.linalg.norm(R) / np.sqrt(M_total)
    main_residual_history.append(residual_norm)
    print(f"Iter {k}: Residual norm (RMSE) = {residual_norm:.4e}")

    # 当前主阶段L2相对误差
    u_pred = np.zeros(len(points))
    u_pred[is_external] = predict(external_points_test, w_out, b_out, alpha_out)
    u_pred[~is_external] = predict(internal_points_test, w_inner, b_inner, alpha_inner)
    u_pred = u_pred.reshape(xx.shape)
    error = u_exact - u_pred
    rel_l2 = np.linalg.norm(error.ravel()) / u_exact_norm
    main_l2_history.append(rel_l2)

    residual_diff = abs(residual_norm - prev_residual) / abs(residual_norm) if residual_norm != 0 else float('inf')
    if residual_diff < delta_threshold and k > 0:
        print(f"Main iteration stopped: residual difference {residual_diff:.4e} < {delta_threshold}")
        break
    prev_residual = residual_norm

    if residual_norm < tol:
        print(f"Main iteration stopped: residual norm {residual_norm:.4e} < tolerance {tol}")
        break

    # 最小二乘更新
    delta_beta, _, _, _ = lstsq(J, -R, cond=1e-12)
    alpha_out += delta_beta[:N]
    alpha_inner += delta_beta[N:]

# 主阶段预测，用于摄动阶段
u_pred_main = np.zeros(len(points))
u_pred_main[is_external] = predict(external_points_test, w_out, b_out, alpha_out)
u_pred_main[~is_external] = predict(internal_points_test, w_inner, b_inner, alpha_inner)
u_pred_main = u_pred_main.reshape(xx.shape)

# 摄动阶段
perturb_residual_history = []
perturb_l2_history = []
epsilon = 1e-4
alpha_out_p = np.zeros(N_perturb)
alpha_inner_p = np.zeros(N_perturb)
max_perturb_iter = 50
perturb_tol = 1e-10

# 更密集点用于摄动阶段
external_points_p, internal_points_p, interface_points_p, boundary_points_p = generate_points(101, 201, 101*4)

prev_perturb_residual = float('inf')
start_time = time.time()

print("\n--- Starting Perturbation Stage ---")
for k in range(max_perturb_iter):
    R_p, J_p = compute_perturb_residual(alpha_out_p, alpha_inner_p, alpha_out, alpha_inner,
                                        external_points_p, internal_points_p, interface_points_p, boundary_points_p)
    M_total_p = len(R_p)
    res_norm = np.linalg.norm(R_p) / np.sqrt(M_total_p)
    print(f"Perturb Iter {k}: Residual (RMSE) = {res_norm:.3e}")
    perturb_residual_history.append(res_norm)

    # 当前总相对L2误差
    u_pred1_current = np.zeros(len(points))
    u_pred1_current[is_external] = predict1(external_points_test, w_out1, b_out1, alpha_out_p)
    u_pred1_current[~is_external] = predict1(internal_points_test, w_inner1, b_inner1, alpha_inner_p)
    u_pred1_current = u_pred1_current.reshape(xx.shape)
    u_total = u_pred_main + epsilon * u_pred1_current
    error_total = u_exact - u_total
    rel_l2_total = np.linalg.norm(error_total.ravel()) / u_exact_norm
    perturb_l2_history.append(rel_l2_total)

    residual_diff = abs(res_norm - prev_perturb_residual) / abs(res_norm) if res_norm != 0 else float('inf')
    if residual_diff < delta_threshold and k > 0:
        print(f"Perturb iteration stopped: residual difference {residual_diff:.4e} < {delta_threshold}")
        break
    prev_perturb_residual = res_norm

    if res_norm < perturb_tol:
        print(f"Perturb iteration stopped: residual norm {res_norm:.4e} < tolerance {perturb_tol}")
        break

    delta_p, _, _, _ = lstsq(J_p, -R_p, cond=1e-14)
    alpha_out_p += delta_p[:N_perturb]
    alpha_inner_p += delta_p[N_perturb:]

end_time = time.time()
elapsed = end_time - start_time
print(f"\nElapsed time for perturbation stage: {elapsed:.6f} seconds")

# 最终预测
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

# 误差统计
max_abs_error = np.max(np.abs(error))
rel_l2_error = np.linalg.norm(error.ravel()) / u_exact_norm
rel_l2_error_stage2 = np.linalg.norm(error1.ravel()) / u_exact_norm
rel_max_error_stage2 = np.max(np.abs(error1)) / np.max(np.abs(u_exact))

print("\n--- Final Error Analysis ---")
print(f"Max absolute error (stage 1): {max_abs_error:.4e}")
print(f"Relative L2 error (stage 1): {rel_l2_error:.4e}")
print(f"Relative L2 error (stage 2): {rel_l2_error_stage2:.4e}")
print(f"Relative max error (stage 2): {rel_max_error_stage2:.4e}")