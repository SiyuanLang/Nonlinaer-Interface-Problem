import numpy as np
from scipy.linalg import lstsq
import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------------ 配置与问题定义 ------------------
T_MAX = 0.2
np.random.seed(42)
N = 500            # 主阶段隐藏节点数
N_PERTURB = 1000   # 摄动阶段节点数
gamma_bc = 1000
gamma_interface = 1000
EPS = 1e-2  # 摄动展开参数，可调小以控制扰动幅度

# 精确解和系数
def u_out_exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y)

def u_inner_exact(x, y, t):
    return t*(x**2 + y**2)

def beta_out(u): return 1.0 + u**2
def beta_inner(u):
    return 1.0 + u

def f_out_exact(x, y, t):
    u = u_out_exact(x,y,t)
    ut = -np.exp(-t)*np.sin(np.pi*x)*np.sin(np.pi*y)
    ux = np.exp(-t)*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
    uy = np.exp(-t)*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
    uxx = -np.exp(-t)*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
    uyy = uxx
    b = beta_out(u); bp = 2*u
    return ut - (b*(uxx+uyy) + bp*(ux**2 + uy**2))

def f_inner_exact(x, y, t):
    u = u_inner_exact(x, y, t)       # t(x² + y²)
    ut = x**2 + y**2
    ux = 2 * t * x
    uy = 2 * t * y
    uxx = 2 * t
    uyy = 2 * t

    b = beta_inner(u)     # 1 + u
    bp = 1.0              # beta_inner'(u) = 1
    return ut - (b * (uxx + uyy) + bp * (ux**2 + uy**2))


def w_jump(x,y,t): return u_out_exact(x,y,t) - u_inner_exact(x,y,t)

def v_jump(x, y, t):
    r = np.sqrt(x**2 + y**2)
    nx, ny = x / r, y / r

    # 外域
    uo = u_out_exact(x, y, t)
    bo = beta_out(uo)    # 1 + uo**2
    uox = np.exp(-t) * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    uoy = np.exp(-t) * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)

    # 内域
    ui = u_inner_exact(x, y, t)
    bi = beta_inner(ui)  # 1 + ui
    uix = 2 * t * x
    uiy = 2 * t * y

    # 法向通量跳跃
    return (
        bo * (uox * nx + uoy * ny)
        - bi * (uix * nx + uiy * ny)
    )

# ------------------ 点生成 ------------------
def generate_points_time_space_mc(N_int, N_ext, N_bd, N_if, N_init, seed=None):
    """
    生成蒙特卡洛采样点，界面半径 r(t) = 1.5*t + 0.5，
    时间范围 [0, T_MAX]。
    返回 (all_ext, all_int, all_if, all_bd, ext_init_pts, int_init_pts)
    """

    # =================== 新增统一界面半径函数 =====================
    def r_interface(t):
        return 1.5 * t + 0.5

    # 随机数发生器（可重复）
    rng = np.random.default_rng(seed)

    # ------------------------
    # 内部工具函数
    # ------------------------
    def sample_region_points(N, region):
        """在整个时域上随机采样时刻 t，并在方形 [-1,1]^2 上采样 (x,y)，
           通过半径比较筛选内/外点。"""
        if N <= 0:
            return np.zeros((0, 3))
        # 外域时刻上限：界面半径<sqrt(2)
        if region == 'outer':
            t_upper = min(T_MAX, (np.sqrt(2.0) - 0.5) / 1.5)  # 解 1.5 t + 0.5 < sqrt(2)
            if t_upper <= 0.0:
                return np.zeros((0, 3))
        else:
            t_upper = T_MAX

        pts_list = []
        need = N
        batch = max(2000, 5 * N)
        while need > 0:
            k = min(batch, 5 * need)
            t = rng.uniform(0.0, t_upper, size=k)
            x = rng.uniform(-1.0, 1.0, size=k)
            y = rng.uniform(-1.0, 1.0, size=k)
            r = r_interface(t)  # MODIFIED: 使用新界面函数
            rr = r * r
            rho2 = x * x + y * y
            if region == 'inner':
                mask = (rho2 <= rr)
            else:
                mask = (rho2 >= rr)
            if np.any(mask):
                xm = x[mask]
                ym = y[mask]
                tm = t[mask]
                take = min(xm.shape[0], need)
                pts_list.append(np.column_stack((xm[:take], ym[:take], tm[:take])))
                need -= take
        return np.vstack(pts_list)

    def sample_region_at_t0(N, region, r0):
        """在 t=0 时刻从方形采样按 r0 划分内外区域"""
        if N <= 0:
            return np.zeros((0, 3))
        pts_list = []
        need = N
        batch = max(2000, 5 * N)
        rr0 = r0 * r0
        while need > 0:
            k = min(batch, 5 * need)
            x = rng.uniform(-1.0, 1.0, size=k)
            y = rng.uniform(-1.0, 1.0, size=k)
            rho2 = x * x + y * y
            if region == 'inner':
                mask = (rho2 <= rr0)
            else:
                mask = (rho2 >= rr0)
            if np.any(mask):
                xm = x[mask]
                ym = y[mask]
                tm = np.zeros(xm.shape[0])  # t=0
                take = min(xm.shape[0], need)
                pts_list.append(np.column_stack((xm[:take], ym[:take], tm[:take])))
                need -= take
        return np.vstack(pts_list)

    def sample_boundary_points(N):
        """在边界 ∂Ω 上等弧长采样"""
        if N <= 0:
            return np.zeros((0, 3))
        t = rng.uniform(0.0, T_MAX, size=N)
        edge = rng.integers(0, 4, size=N)   # 0:bottom,1:right,2:top,3:left
        u = rng.uniform(0.0, 1.0, size=N)   # 边上参数
        x = np.empty(N)
        y = np.empty(N)
        m = (edge == 0)  # bottom
        x[m] = -1.0 + 2.0 * u[m]
        y[m] = -1.0
        m = (edge == 1)  # right
        x[m] = 1.0
        y[m] = -1.0 + 2.0 * u[m]
        m = (edge == 2)  # top
        x[m] = 1.0 - 2.0 * u[m]
        y[m] = 1.0
        m = (edge == 3)  # left
        x[m] = -1.0
        y[m] = 1.0 - 2.0 * u[m]
        return np.column_stack((x, y, t))

    def sample_interface_points(N):
        """采样界面上的点"""
        if N <= 0:
            return np.zeros((0, 3))
        # 界面存在时间范围
        t_if_max = T_MAX  # 因为0.2时半径0.8<1，所以全范围可用
        t = rng.uniform(0.0, t_if_max, size=N)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=N)
        r = r_interface(t)  # MODIFIED
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y, t))

    # ------------------------
    # 初始时刻 t=0 的内/外点
    # ------------------------
    r0 = r_interface(0.0)  # 新界面函数
    p_inner0 = (np.pi * r0 * r0) / 4.0  # 初始内域面积比例
    N_init_inner = int(np.round(N_init * p_inner0))
    N_init_outer = max(0, N_init - N_init_inner)

    int_init_pts = sample_region_at_t0(N_init_inner, 'inner', r0)
    ext_init_pts = sample_region_at_t0(N_init_outer, 'outer', r0)

    # ------------------------
    # 时空蒙特卡洛样本
    # ------------------------
    ext_pts = sample_region_points(N_ext, 'outer')
    int_pts = sample_region_points(N_int, 'inner')
    bd_pts  = sample_boundary_points(N_bd)
    if_pts  = sample_interface_points(N_if)

    # ------------------------
    # 输出组装
    # ------------------------
    all_ext = [ext_init_pts, ext_pts] if ext_pts.size > 0 else [ext_init_pts]
    all_int = [int_init_pts, int_pts] if int_pts.size > 0 else [int_init_pts]
    all_if  = [if_pts] if if_pts.size > 0 else [np.zeros((0, 3))]
    all_bd  = [bd_pts] if bd_pts.size > 0 else [np.zeros((0, 3))]

    return (np.vstack(all_ext), np.vstack(all_int),
            np.vstack(all_if), np.vstack(all_bd),
            ext_init_pts, int_init_pts)
# ------------------ 初始化权重 ------------------
w_out = np.random.randn(N, 3); b_out = 0.1*np.random.randn(N)
w_inner = np.random.randn(N, 3); b_inner = 0.1*np.random.randn(N)
# 二阶段sin激活
w_out1 = np.random.randn(N_PERTURB, 3); b_out1 = 0.1*np.random.randn(N_PERTURB)
w_inner1 = np.random.randn(N_PERTURB, 3); b_inner1 = 0.1*np.random.randn(N_PERTURB)

def compute_residual_and_jacobian(
    alpha_out, alpha_inner,    # 网络系数
    ext_pts, int_pts, if_pts, bd_pts,   # 原采样点
    ext_init_pts, int_init_pts, gamma_init=1000 # 初始点和惩罚系数
):
    ...
    """
    主阶段残差与严格雅可比（x, y, t 三维输入）
    """
    M_out = len(ext_pts)
    M_in = len(int_pts)
    M_if = len(if_pts)
    M_bd = len(bd_pts)
    M_init_ext = len(ext_init_pts) if ext_init_pts is not None else 0
    M_init_int = len(int_init_pts) if int_init_pts is not None else 0
    M_total = M_out + M_in + 2*M_if + M_bd + M_init_ext + M_init_int

    # 分配残差和雅可比
    R = np.zeros(M_total)
    J = np.zeros((M_total, 2*N))

    # === 外部 PDE 点 ===
    for j,(x,y,t) in enumerate(ext_pts):
        z = w_out[:,0]*x + w_out[:,1]*y + w_out[:,2]*t + b_out
        phi = np.tanh(z)
        phi_p = 1 - phi**2      # 一阶导因子
        phi_pp = -2*phi*(phi_p) # 二阶导因子
        u = np.dot(alpha_out, phi)
        ut = np.dot(alpha_out, w_out[:,2]*phi_p)
        ux = np.dot(alpha_out, w_out[:,0]*phi_p)
        uy = np.dot(alpha_out, w_out[:,1]*phi_p)
        uxx = np.dot(alpha_out, (w_out[:,0]**2)*phi_pp)
        uyy = np.dot(alpha_out, (w_out[:,1]**2)*phi_pp)
        bval = beta_out(u)
        bp = 2*u
        bpp = 2.0
        fval = f_out_exact(x, y, t)

        Rj = ut - ( bval*(uxx+uyy) + bp*(ux**2+uy**2) ) - fval
        R[j] = Rj

        for k in range(N):
            du_dak   = phi[k]
            dux_dak  = w_out[k,0]*phi_p[k]
            duy_dak  = w_out[k,1]*phi_p[k]
            dut_dak  = w_out[k,2]*phi_p[k]
            duxx_dak = (w_out[k,0]**2)*phi_pp[k]
            duyy_dak = (w_out[k,1]**2)*phi_pp[k]

            dR_dak = dut_dak \
                     - ( bp*du_dak       * (uxx+uyy)
                         + bval*(duxx_dak + duyy_dak)
                         + bpp*du_dak     *(ux**2+uy**2)
                         + bp*( 2*ux*dux_dak + 2*uy*duy_dak ) )
            J[j,k] = dR_dak

    # === 内部 PDE 点 ===
    for idx,(x,y,t) in enumerate(int_pts):
        j = M_out + idx
        z = w_inner[:,0]*x + w_inner[:,1]*y + w_inner[:,2]*t + b_inner
        phi = np.tanh(z)
        phi_p = 1 - phi**2
        phi_pp = -2*phi*(phi_p)
        u = np.dot(alpha_inner, phi)
        ut = np.dot(alpha_inner, w_inner[:,2]*phi_p)
        ux = np.dot(alpha_inner, w_inner[:,0]*phi_p)
        uy = np.dot(alpha_inner, w_inner[:,1]*phi_p)
        uxx = np.dot(alpha_inner, (w_inner[:,0]**2)*phi_pp)
        uyy = np.dot(alpha_inner, (w_inner[:,1]**2)*phi_pp)
        bval = beta_inner(u)
        bp = 1.0
        bpp = 0.0
        fval = f_inner_exact(x, y, t)

        Rj = ut - ( bval*(uxx+uyy) + bp*(ux**2+uy**2) ) - fval
        R[j] = Rj

        for k in range(N):
            du_dak   = phi[k]
            dux_dak  = w_inner[k,0]*phi_p[k]
            duy_dak  = w_inner[k,1]*phi_p[k]
            dut_dak  = w_inner[k,2]*phi_p[k]
            duxx_dak = (w_inner[k,0]**2)*phi_pp[k]
            duyy_dak = (w_inner[k,1]**2)*phi_pp[k]
            dR_dak = dut_dak \
                     - ( bp*du_dak*(uxx+uyy)
                         + bval*(duxx_dak+duyy_dak)
                         + bpp*du_dak*(ux**2+uy**2)
                         + bp*(2*ux*dux_dak + 2*uy*duy_dak) )
            J[j,N+k] = dR_dak

    # === 界面条件1: [u] = w_jump ===
    for idx,(x,y,t) in enumerate(if_pts):
        j = M_out+M_in+idx
        phi_out = np.tanh(w_out[:,0]*x + w_out[:,1]*y + w_out[:,2]*t + b_out)
        phi_in  = np.tanh(w_inner[:,0]*x + w_inner[:,1]*y + w_inner[:,2]*t + b_inner)
        u_outv = np.dot(alpha_out, phi_out)
        u_inv  = np.dot(alpha_inner, phi_in)
        Rj = (u_outv - u_inv - w_jump(x,y,t))*gamma_interface
        R[j] = Rj
        J[j,:N] = phi_out*gamma_interface
        J[j,N:] = -phi_in*gamma_interface

    # === 界面条件2: [β ∂u/∂n] = v_jump ===
    for idx,(x,y,t) in enumerate(if_pts):
        j = M_out+M_in+M_if+idx
        r = np.sqrt(x**2+y**2)
        nx, ny = x/r, y/r
        # 外部
        z_out = w_out[:,0]*x + w_out[:,1]*y + w_out[:,2]*t + b_out
        phi_out = np.tanh(z_out)
        phi_p_out = 1 - phi_out**2
        u_outv = np.dot(alpha_out, phi_out)
        ux_outv = np.dot(alpha_out, w_out[:,0]*phi_p_out)
        uy_outv = np.dot(alpha_out, w_out[:,1]*phi_p_out)
        gradn_out = ux_outv*nx + uy_outv*ny
        b_outv = beta_out(u_outv)
        bp_out = 2*u_outv
        # 内部
        z_in = w_inner[:,0]*x + w_inner[:,1]*y + w_inner[:,2]*t + b_inner
        phi_in = np.tanh(z_in)
        phi_p_in = 1 - phi_in**2
        u_inv = np.dot(alpha_inner, phi_in)
        ux_inv = np.dot(alpha_inner, w_inner[:,0]*phi_p_in)
        uy_inv = np.dot(alpha_inner, w_inner[:,1]*phi_p_in)
        gradn_in = ux_inv*nx + uy_inv*ny
        b_inv = beta_inner(u_inv)
        bp_in = 1.0

        Rj = (b_outv*gradn_out - b_inv*gradn_in - v_jump(x,y,t))*gamma_interface
        R[j] = Rj

        # jac 外部
        for k in range(N):
            du_dak = phi_out[k]
            dux_dak = w_out[k,0]*phi_p_out[k]
            duy_dak = w_out[k,1]*phi_p_out[k]
            dgradn_dak = dux_dak*nx + duy_dak*ny
            J[j,k] = (bp_out*du_dak*gradn_out + b_outv*dgradn_dak)*gamma_interface
        # jac 内部
        for k in range(N):
            du_dak = phi_in[k]
            dux_dak = w_inner[k,0]*phi_p_in[k]
            duy_dak = w_inner[k,1]*phi_p_in[k]
            dgradn_dak = dux_dak*nx + duy_dak*ny
            J[j,N+k] = -(bp_in*du_dak*gradn_in + b_inv*dgradn_dak)*gamma_interface

    # === 边界条件 ===
    for idx,(x,y,t) in enumerate(bd_pts):
        j = M_out+M_in+2*M_if+idx
        phi_out = np.tanh(w_out[:,0]*x + w_out[:,1]*y + w_out[:,2]*t + b_out)
        u_pred = np.dot(alpha_out, phi_out)
        Rj = (u_pred - u_out_exact(x,y,t))*gamma_bc
        R[j] = Rj
        J[j,:N] = phi_out*gamma_bc

    if ext_init_pts is not None:
        for idx, (x, y, t) in enumerate(ext_init_pts):
            j = M_out + M_in + 2 * M_if + M_bd + idx  # 正确累加偏移
            phi_out = np.tanh(w_out[:, 0] * x + w_out[:, 1] * y + w_out[:, 2] * t + b_out)
            u_pred = np.dot(alpha_out, phi_out)
            R[j] = (u_pred - u_out_exact(x, y, t)) * gamma_init
            J[j, :N] = phi_out * gamma_init

        # === 初始条件：内域（t=0，r < r(0)）===
    if int_init_pts is not None:
        offset = M_out + M_in + 2 * M_if + M_bd + M_init_ext
        for idx, (x, y, t) in enumerate(int_init_pts):
            j = offset + idx
            phi_in = np.tanh(w_inner[:, 0] * x + w_inner[:, 1] * y + w_inner[:, 2] * t + b_inner)
            u_pred = np.dot(alpha_inner, phi_in)
            R[j] = (u_pred - u_inner_exact(x, y, t)) * gamma_init
            J[j, N:] = phi_in * gamma_init

    return R, J

def compute_perturb_residual_and_jacobian_sin(
    dalpha_out, dalpha_in,
    alpha_out_bg, alpha_in_bg,
    ext_pts, int_pts, if_pts, bd_pts,
    ext_init_pts, int_init_pts, gamma_init=1000
):
    # 二阶段摄动展开残差与雅可比 (tanh 基函数, 明确 ε² 截断)
    # dalpha_out, dalpha_in: 摄动阶段权重 (δ α)
    # alpha_out_bg, alpha_in_bg: 背景解权重 (α 主阶段训练好的)
    # ext_pts, int_pts, if_pts, bd_pts: 各子域采样点
    M_out, M_in, M_if, M_bd = len(ext_pts), len(int_pts), len(if_pts), len(bd_pts)
    M_init_ext = len(ext_init_pts) if ext_init_pts is not None else 0
    M_init_int = len(int_init_pts) if int_init_pts is not None else 0
    M_total = M_out + M_in + 2 * M_if + M_bd + M_init_ext + M_init_int

    R = np.zeros(M_total)
    J = np.zeros((M_total, 2 * N_PERTURB))

    # ---------------- 外部 PDE ----------------
    for j, (x, y, t) in enumerate(ext_pts):
        # 背景解 (tanh, 及其各阶导)
        zb = w_out[:, 0] * x + w_out[:, 1] * y + w_out[:, 2] * t + b_out
        phi0 = np.tanh(zb)
        phi0_p = 1 - phi0 ** 2
        phi0_pp = -2 * phi0 * phi0_p
        u0 = np.dot(alpha_out_bg, phi0)
        ut0 = np.dot(alpha_out_bg, w_out[:, 2] * phi0_p)
        ux0 = np.dot(alpha_out_bg, w_out[:, 0] * phi0_p)
        uy0 = np.dot(alpha_out_bg, w_out[:, 1] * phi0_p)
        uxx0 = np.dot(alpha_out_bg, (w_out[:, 0] ** 2) * phi0_pp)
        uyy0 = np.dot(alpha_out_bg, (w_out[:, 1] ** 2) * phi0_pp)

        # 摄动解 (tanh 及其导)
        zp = w_out1[:, 0] * x + w_out1[:, 1] * y + w_out1[:, 2] * t + b_out1
        tp = np.tanh(zp)
        tp_p = 1 - tp ** 2
        tp_pp = -2 * tp * tp_p
        u1 = np.dot(dalpha_out, tp)
        ut1 = np.dot(dalpha_out, w_out1[:, 2] * tp_p)
        ux1 = np.dot(dalpha_out, w_out1[:, 0] * tp_p)
        uy1 = np.dot(dalpha_out, w_out1[:, 1] * tp_p)
        uxx1 = np.dot(dalpha_out, (w_out1[:, 0] ** 2) * tp_pp)
        uyy1 = np.dot(dalpha_out, (w_out1[:, 1] ** 2) * tp_pp)

        # 各阶β和空间微分（外域：β(u)=1+u^2, β'=2u, β''=2, β'''=0）
        β0 = beta_out(u0)
        βp0 = 2 * u0
        βpp0 = 2.0
        β3 = 0.0

        grad0_sq = ux0 ** 2 + uy0 ** 2
        grad1_dot0 = ux0 * ux1 + uy0 * uy1
        grad1_sq = ux1 ** 2 + uy1 ** 2
        lap0 = uxx0 + uyy0
        lap1 = uxx1 + uyy1

        # 残差（严格展开到 ε²）
        res = ut0 + EPS * ut1 \
              - (
                      β0 * lap0
                      + βp0 * grad0_sq

                      + EPS * (
                              β0 * lap1
                              + 2 * βp0 * grad1_dot0
                              + βp0 * u1 * lap0
                              + βpp0 * u1 * grad0_sq
                      )
                      + EPS ** 2 * (
                              βp0 * u1 * lap1
                              + 2 * βpp0 * u1 * grad1_dot0
                              + βp0 * grad1_sq
                              + 0.5 * βpp0 * u1 ** 2 * lap0
                              + 0.5 * β3 * u1 ** 2 * grad0_sq
                      )
              ) - f_out_exact(x, y, t)
        R[j] = res / EPS

        # --------- 雅可比 w.r.t δα_out[k] (严格展开,对每个扰动参数) -------------
        for k in range(N_PERTURB):
            zk = w_out1[k, 0] * x + w_out1[k, 1] * y + w_out1[k, 2] * t + b_out1[k]
            tk = np.tanh(zk)
            tk_p = 1 - tk ** 2
            tk_pp = -2 * tk * tk_p
            du = tk
            dut = w_out1[k, 2] * tk_p
            dux = w_out1[k, 0] * tk_p
            duy = w_out1[k, 1] * tk_p
            duxx = (w_out1[k, 0] ** 2) * tk_pp
            duyy = (w_out1[k, 1] ** 2) * tk_pp

            dugrad0 = ux0 * dux + uy0 * duy  # ∇u_0·∇φ_k
            dugrad1 = ux1 * dux + uy1 * duy  # ∇u_1·∇φ_k
            dlap1 = duxx + duyy

            jac = dut * EPS - \
            (
                    EPS * (
                    β0 * dlap1
                    + 2 * βp0 * dugrad0
                    + βp0 * du * lap0
                    + βpp0 * du * grad0_sq
                    )
                    + EPS ** 2 * (
                            βp0 * du * lap1
                            + βp0 * u1 * dlap1
                            + 2 * βpp0 * (du * grad1_dot0 + u1 * dugrad0)
                            + βp0 * 2 * (ux1 * dux + uy1 * duy)  # d|∇u_1|^2 = 2∇u_1·∇φ_k
                            + u1 * du * β3 * grad0_sq
                            + βpp0 * u1 * du * lap0
                    )
            )
            J[j, k] = jac / EPS

    # ---------------- 内部 PDE ----------------
    for idx, (x, y, t) in enumerate(int_pts):
        j = M_out + idx
        # 背景解 (tanh)
        zb = w_inner[:, 0] * x + w_inner[:, 1] * y + w_inner[:, 2] * t + b_inner
        phi0 = np.tanh(zb)
        phi0_p = 1 - phi0 ** 2
        phi0_pp = -2 * phi0 * phi0_p
        u0 = np.dot(alpha_in_bg, phi0)
        ut0 = np.dot(alpha_in_bg, w_inner[:, 2] * phi0_p)
        ux0 = np.dot(alpha_in_bg, w_inner[:, 0] * phi0_p)
        uy0 = np.dot(alpha_in_bg, w_inner[:, 1] * phi0_p)
        uxx0 = np.dot(alpha_in_bg, (w_inner[:, 0] ** 2) * phi0_pp)
        uyy0 = np.dot(alpha_in_bg, (w_inner[:, 1] ** 2) * phi0_pp)

        # 摄动解 (tanh)
        zp = w_inner1[:, 0] * x + w_inner1[:, 1] * y + w_inner1[:, 2] * t + b_inner1
        tp = np.tanh(zp)
        tp_p = 1 - tp ** 2
        tp_pp = -2 * tp * tp_p
        u1 = np.dot(dalpha_in, tp)
        ut1 = np.dot(dalpha_in, w_inner1[:, 2] * tp_p)
        ux1 = np.dot(dalpha_in, w_inner1[:, 0] * tp_p)
        uy1 = np.dot(dalpha_in, w_inner1[:, 1] * tp_p)
        uxx1 = np.dot(dalpha_in, (w_inner1[:, 0] ** 2) * tp_pp)
        uyy1 = np.dot(dalpha_in, (w_inner1[:, 1] ** 2) * tp_pp)

        # 内域 β 相关（保持与原代码一致：β'(u)=1, β''(u)=0）
        β0 = beta_inner(u0)
        βp0 = 1.0
        βpp0 = 0.0
        β3 = 0.0

        grad0_sq = ux0 ** 2 + uy0 ** 2
        grad1_dot0 = ux0 * ux1 + uy0 * uy1
        grad1_sq = ux1 ** 2 + uy1 ** 2
        lap0 = uxx0 + uyy0
        lap1 = uxx1 + uyy1

        # 残差
        res = ut0 + EPS * ut1 \
              - (
                      β0 * lap0
                      + βp0 * grad0_sq
                      + EPS * (
                              β0 * lap1
                              + 2 * βp0 * grad1_dot0
                              + βp0 * u1 * lap0
                              + βpp0 * u1 * grad0_sq
                      )
                      + EPS ** 2 * (
                              βp0 * u1 * lap1
                              + 2 * βpp0 * u1 * grad1_dot0
                              + βp0 * grad1_sq
                              + 0.5 * βpp0 * u1 ** 2 * lap0
                              + 0.5 * β3 * u1 ** 2 * grad0_sq
                      )
              ) - f_inner_exact(x, y, t)
        R[j] = res / EPS

        # ------ 内部域Jacobi, 对dalpha_in
        for k in range(N_PERTURB):
            zk = w_inner1[k, 0] * x + w_inner1[k, 1] * y + w_inner1[k, 2] * t + b_inner1[k]
            tk = np.tanh(zk)
            tk_p = 1 - tk ** 2
            tk_pp = -2 * tk * tk_p
            du = tk
            dut = w_inner1[k, 2] * tk_p
            dux = w_inner1[k, 0] * tk_p
            duy = w_inner1[k, 1] * tk_p
            duxx = (w_inner1[k, 0] ** 2) * tk_pp
            duyy = (w_inner1[k, 1] ** 2) * tk_pp

            dugrad0 = ux0 * dux + uy0 * duy
            dugrad1 = ux1 * dux + uy1 * duy
            dlap1 = duxx + duyy

            jac = dut * EPS
            jac -= (
                    EPS * (
                    β0 * dlap1
                    + 2 * βp0 * dugrad0
                    + βp0 * du * lap0
                    + βpp0 * du * grad0_sq
            )
                    + EPS ** 2 * (
                            βp0 * du * lap1
                            + βp0 * u1 * dlap1
                            + 2 * βpp0 * (du * grad1_dot0 + u1 * dugrad0)
                            + βp0 * 2 * (ux1 * dux + uy1 * duy)
                            + u1 * du * β3 * grad0_sq
                            + βpp0 * u1 * du * lap0
                    )
            )
            J[j, N_PERTURB + k] = jac / EPS

    # ---------------- 界面条件 1: 值跳 ----------------
    for idx,(x,y,t) in enumerate(if_pts):
        j = M_out + M_in + idx
        u_out_bg = np.tanh(w_out[:,0]*x + w_out[:,1]*y + w_out[:,2]*t + b_out) @ alpha_out_bg
        u_out_p  = np.tanh(w_out1[:,0]*x + w_out1[:,1]*y + w_out1[:,2]*t + b_out1) @ dalpha_out
        u_in_bg  = np.tanh(w_inner[:,0]*x + w_inner[:,1]*y + w_inner[:,2]*t + b_inner) @ alpha_in_bg
        u_in_p   = np.tanh(w_inner1[:,0]*x + w_inner1[:,1]*y + w_inner1[:,2]*t + b_inner1) @ dalpha_in
        R[j] = (u_out_bg + EPS*u_out_p - u_in_bg - EPS*u_in_p - w_jump(x,y,t))/EPS*gamma_interface

        phi_p_out = np.tanh(w_out1[:,0]*x + w_out1[:,1]*y + w_out1[:,2]*t + b_out1)
        phi_p_in  = np.tanh(w_inner1[:,0]*x + w_inner1[:,1]*y + w_inner1[:,2]*t + b_inner1)
        J[j,:N_PERTURB]   = phi_p_out*gamma_interface
        J[j,N_PERTURB:]   = -phi_p_in*gamma_interface

    # ---------------- 界面条件 2: 法向导数跳 ----------------
    for idx, (x, y, t) in enumerate(if_pts):
        j = M_out + M_in + M_if + idx
        r = np.sqrt(x ** 2 + y ** 2)
        nx, ny = x / r, y / r

        # --- Out (外域) ---
        # 主解
        z0o = w_out[:, 0] * x + w_out[:, 1] * y + w_out[:, 2] * t + b_out
        phi0_out = np.tanh(z0o)
        phi0_p_out = 1 - phi0_out ** 2
        u0o = alpha_out_bg @ phi0_out
        ux0o = alpha_out_bg @ (w_out[:, 0] * phi0_p_out)
        uy0o = alpha_out_bg @ (w_out[:, 1] * phi0_p_out)
        gradn0o = ux0o * nx + uy0o * ny

        # 摄动 (tanh)
        z1o = w_out1[:, 0] * x + w_out1[:, 1] * y + w_out1[:, 2] * t + b_out1
        phi1_out = np.tanh(z1o)
        phi1_p_out = 1 - phi1_out ** 2
        phi1_pp_out = -2 * phi1_out * phi1_p_out
        u1o = dalpha_out @ phi1_out
        ux1o = dalpha_out @ (w_out1[:, 0] * phi1_p_out)
        uy1o = dalpha_out @ (w_out1[:, 1] * phi1_p_out)
        gradn1o = ux1o * nx + uy1o * ny

        beta0o = beta_out(u0o)
        beta0p_o = 2 * u0o
        beta0pp_o = 2.0

        # --- In (内域) ---
        # 主解
        z0i = w_inner[:, 0] * x + w_inner[:, 1] * y + w_inner[:, 2] * t + b_inner
        phi0_in = np.tanh(z0i)
        phi0_p_in = 1 - phi0_in ** 2
        u0i = alpha_in_bg @ phi0_in
        ux0i = alpha_in_bg @ (w_inner[:, 0] * phi0_p_in)
        uy0i = alpha_in_bg @ (w_inner[:, 1] * phi0_p_in)
        gradn0i = ux0i * nx + uy0i * ny

        # 摄动 (tanh)
        z1i = w_inner1[:, 0] * x + w_inner1[:, 1] * y + w_inner1[:, 2] * t + b_inner1
        phi1_in = np.tanh(z1i)
        phi1_p_in = 1 - phi1_in ** 2
        phi1_pp_in = -2 * phi1_in * phi1_p_in
        u1i = dalpha_in @ phi1_in
        ux1i = dalpha_in @ (w_inner1[:, 0] * phi1_p_in)
        uy1i = dalpha_in @ (w_inner1[:, 1] * phi1_p_in)
        gradn1i = ux1i * nx + uy1i * ny

        beta0i = beta_inner(u0i)
        beta0p_i = 1.0
        beta0pp_i = 0.0

        # ε⁰项
        term_o_0 = beta0o * gradn0o
        term_i_0 = beta0i * gradn0i
        # ε¹项
        term_o_1 = beta0o * gradn1o + beta0p_o * u1o * gradn0o
        term_i_1 = beta0i * gradn1i + beta0p_i * u1i * gradn0i
        # ε²项
        term_o_2 = beta0p_o * u1o * gradn1o + 0.5 * beta0pp_o * u1o ** 2 * gradn0o
        term_i_2 = beta0p_i * u1i * gradn1i + 0.5 * beta0pp_i * u1i ** 2 * gradn0i

        flux_jump = term_o_0 - term_i_0 \
                    + EPS * (term_o_1 - term_i_1) \
                    + EPS ** 2 * (term_o_2 - term_i_2)
        R[j] = (flux_jump - v_jump(x, y, t))/EPS * gamma_interface

        # --- Out，对 δα_out ---
        for k in range(N_PERTURB):
            zk = w_out1[k, 0] * x + w_out1[k, 1] * y + w_out1[k, 2] * t + b_out1[k]
            tk = np.tanh(zk)
            tk_p = 1 - tk ** 2
            du1 = tk
            dux1 = w_out1[k, 0] * tk_p
            duy1 = w_out1[k, 1] * tk_p
            dgradn1 = dux1 * nx + duy1 * ny

            # ∂[ε^1~]项 wrt dalpha_out[k]
            dterm_o_1 = beta0o * dgradn1 + beta0p_o * du1 * gradn0o
            # ∂[ε^2~]项 wrt dalpha_out[k]
            dterm_o_2 = beta0p_o * (du1 * gradn1o + u1o * dgradn1) \
                        + beta0pp_o * u1o * du1 * gradn0o
            # 合计
            J[j, k] = (dterm_o_1 + EPS * dterm_o_2) * gamma_interface

        # --- In，对 δα_in ---
        for k in range(N_PERTURB):
            zk = w_inner1[k, 0] * x + w_inner1[k, 1] * y + w_inner1[k, 2] * t + b_inner1[k]
            tk = np.tanh(zk)
            tk_p = 1 - tk ** 2
            du1 = tk
            dux1 = w_inner1[k, 0] * tk_p
            duy1 = w_inner1[k, 1] * tk_p
            dgradn1 = dux1 * nx + duy1 * ny

            dterm_i_1 = beta0i * dgradn1 + beta0p_i * du1 * gradn0i
            dterm_i_2 = beta0p_i * (du1 * gradn1i + u1i * dgradn1) + beta0pp_i * u1i * du1 * gradn0i
            J[j, N_PERTURB + k] = (- dterm_i_1 - EPS * dterm_i_2) * gamma_interface

    # ---------------- 边界条件 ----------------
    for idx,(x,y,t) in enumerate(bd_pts):
        j = M_out + M_in + 2*M_if + idx
        ubg = np.tanh(w_out[:,0]*x + w_out[:,1]*y + w_out[:,2]*t + b_out) @ alpha_out_bg
        phi1 = np.tanh(w_out1[:,0]*x + w_out1[:,1]*y + w_out1[:,2]*t + b_out1)
        u_pred = ubg + EPS*(phi1 @ dalpha_out)
        R[j] = (u_pred - u_out_exact(x,y,t))/EPS*gamma_bc
        J[j,:N_PERTURB] = phi1*gamma_bc

    # ---------------- 初始条件 ----------------
    if ext_init_pts is not None:
        for idx, (x, y, t) in enumerate(ext_init_pts):
            j_ind = M_out + M_in + 2 * M_if + M_bd + idx
            zb = w_out[:, 0] * x + w_out[:, 1] * y + w_out[:, 2] * t + b_out
            u0 = alpha_out_bg @ np.tanh(zb)
            zp = w_out1[:, 0] * x + w_out1[:, 1] * y + w_out1[:, 2] * t + b_out1
            phi1 = np.tanh(zp)
            u1 = dalpha_out @ phi1

            R[j_ind] = (u0 + EPS * u1 - u_out_exact(x, y, t))/EPS * gamma_init
            J[j_ind, :N_PERTURB] = phi1 * gamma_init  # 仅影响外部dalpha_out分量

    if int_init_pts is not None:
        offset = M_out + M_in + 2 * M_if + M_bd + (len(ext_init_pts) if ext_init_pts is not None else 0)
        for idx, (x, y, t) in enumerate(int_init_pts):
            j_ind = offset + idx
            zb = w_inner[:, 0] * x + w_inner[:, 1] * y + w_inner[:, 2] * t + b_inner
            u0 = alpha_in_bg @ np.tanh(zb)
            zp = w_inner1[:, 0] * x + w_inner1[:, 1] * y + w_inner1[:, 2] * t + b_inner1
            phi1 = np.tanh(zp)
            u1 = dalpha_in @ phi1

            R[j_ind] = (u0 + EPS * u1 - u_inner_exact(x, y, t))/EPS * gamma_init
            J[j_ind, N_PERTURB:] = phi1 * gamma_init  # 仅影响内部dalpha_in分量

    return R, J

# ------------------ 误差计算 ------------------
def _iter_space_time_grid(T_MAX, nx=30, ny=30, nt=5):
    # 生成计算误差使用的时空采样点
    t_vals = np.linspace(0, T_MAX, nt)
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y)
    # 展平成 (M,) 方便索引
    xr, yr = xx.ravel(), yy.ravel()
    # 预先打包纯空间网格（不含时间），每个时间步再拼 t
    base_xy = np.column_stack((xr, yr))  # (M,2)
    return t_vals, xr, yr, base_xy

def _relative_error_core(
    T_MAX,
    # 预测器：分别对外部/内部区域给出预测值，接口如下
    pred_out_fn,    # (pts: (M,3)) -> upred_out: (M,)
    pred_in_fn,     # (pts: (M,3)) -> upred_in : (M,)
    # 精确解
    out_exact=u_out_exact,
    in_exact=u_inner_exact,
    # 采样分辨率
    nx=30, ny=30, nt=5
):
    # 累积外、内域的误差
    err_out_num = 0.0
    err_out_den = 0.0
    err_in_num  = 0.0
    err_in_den  = 0.0

    # 生成网格
    t_vals, xr, yr, base_xy = _iter_space_time_grid(T_MAX, nx, ny, nt)
    M = base_xy.shape[0]

    for t in t_vals:
        # 半径随时间变化：rt = 0.5*t + 0.5
        rt = 0.5*t + 0.5
        # 组装 (x,y,t)
        pts = np.column_stack((base_xy, np.full(M, t)))
        # 外部/内部掩码
        mask_out = (xr**2 + yr**2) >= (rt**2)
        mask_in  = ~mask_out

        # 外部
        if np.any(mask_out):
            mpts = pts[mask_out]                            # (Mo,3)
            upred = pred_out_fn(mpts)                       # (Mo,)
            uex   = out_exact(mpts[:,0], mpts[:,1], mpts[:,2])
            err_out_num += np.sum((upred - uex)**2)
            err_out_den += np.sum(uex**2)

        # 内部
        if np.any(mask_in):
            mpts = pts[mask_in]                             # (Mi,3)
            upred = pred_in_fn(mpts)                        # (Mi,)
            uex   = in_exact(mpts[:,0], mpts[:,1], mpts[:,2])
            err_in_num += np.sum((upred - uex)**2)
            err_in_den += np.sum(uex**2)

    # 返回相对误差
    rel_out = np.sqrt(err_out_num/err_out_den) if err_out_den > 0 else np.nan
    rel_in  = np.sqrt(err_in_num /err_in_den ) if err_in_den  > 0 else np.nan
    return rel_out, rel_in

def compute_relative_error_tanh(
    alpha_out, alpha_inner,
    w_out, b_out, w_inner, b_inner,
    T_MAX,
    out_exact=u_out_exact,
    in_exact=u_inner_exact,
    nx=30, ny=30, nt=5
):
    # 构造仅含 tanh 的预测器（外部）
    def pred_out_fn(mpts):
        # mpts: (M,3), 取前两列与时间一并送入线性层
        z = mpts @ w_out.T + b_out       # (M, N_out)
        return np.tanh(z) @ alpha_out    # (M,)

    # 构造仅含 tanh 的预测器（内部）
    def pred_in_fn(mpts):
        z = mpts @ w_inner.T + b_inner   # (M, N_in)
        return np.tanh(z) @ alpha_inner  # (M,)

    return _relative_error_core(
        T_MAX,
        pred_out_fn, pred_in_fn,
        out_exact=out_exact, in_exact=in_exact,
        nx=nx, ny=ny, nt=nt
    )

def compute_relative_error_perturbed(
    alpha_out_bg, alpha_in_bg,      # 主解权重
    dalpha_out, dalpha_in,          # 摄动权重
    EPS,                             # 摄动幅度
    w_out, b_out, w_inner, b_inner, # 主解网络参数
    w_out1, b_out1, w_inner1, b_inner1, # 摄动网络参数
    T_MAX,
    out_exact=u_out_exact,
    in_exact=u_inner_exact,
    nx=30, ny=30, nt=5
):
    # 外部：tanh 主项 + EPS * sin 摄动项
    def pred_out_fn(mpts):
        z0 = mpts @ w_out.T  + b_out     # (M, N0)
        u0 = np.tanh(z0) @ alpha_out_bg  # (M,)
        z1 = mpts @ w_out1.T + b_out1    # (M, N1)
        u1 = np.tanh(z1) @ dalpha_out     # (M,)
        return u0 + EPS*u1

    # 内部：tanh 主项 + EPS * sin 摄动项
    def pred_in_fn(mpts):
        z0 = mpts @ w_inner.T  + b_inner
        u0 = np.tanh(z0) @ alpha_in_bg
        z1 = mpts @ w_inner1.T + b_inner1
        u1 = np.tanh(z1) @ dalpha_in
        return u0 + EPS*u1

    return _relative_error_core(
        T_MAX,
        pred_out_fn, pred_in_fn,
        out_exact=out_exact, in_exact=in_exact,
        nx=nx, ny=ny, nt=nt
    )

# ------------------ 主程序 ------------------
if __name__ == "__main__":
    # 提高采样点分辨率
    # N_space = 51
    # Mb, N_time = pick_Mb_Ntime(N_space, T_MAX)
    # N_interface = pick_Ninterface(N_space, T_MAX)
    # print(N_space, N_interface, Mb, N_time)
    # ext_pts, int_pts, if_pts, bd_pts, ext_init_pts, int_init_pts = generate_points_time_space(N_space, N_interface, Mb, 11)
    alpha_out = np.zeros(N)
    alpha_inner = np.zeros(N)

    main_residual_history = []
    main_l2_out_history = []
    main_l2_in_history = []

    for it in range(7):  # 根据需要调整迭代数
        ext_pts, int_pts, if_pts, bd_pts, ext_init_pts, int_init_pts = generate_points_time_space_mc(4000, 6095, 320, 208, 1000)
        R, J = compute_residual_and_jacobian(alpha_out, alpha_inner,
                                             ext_pts, int_pts, if_pts, bd_pts, ext_init_pts, int_init_pts)
        res_norm = np.linalg.norm(R) / np.sqrt(len(R))
        if res_norm>1:
            cond0 = res_norm * 1e-14
        else:
            cond0 = 0.0

        delta, _, _, _ = lstsq(J, -R, cond=1e-12)
        alpha_out += delta[:N]
        alpha_inner += delta[N:]
        rel_out, rel_in = compute_relative_error_tanh(alpha_out, alpha_inner,w_out, b_out, w_inner, b_inner,
    T_MAX,
    out_exact=u_out_exact,
    in_exact=u_inner_exact,)
        main_residual_history.append(res_norm)
        main_l2_out_history.append(rel_out)
        main_l2_in_history.append(rel_in)
        print(f"[Main {it}] Res={res_norm:.3e} RelErr_out={rel_out:.3e} RelErr_in={rel_in:.3e}")

    # ----- Stage 1结束后，准备Stage 2 -----
    dalpha_out = np.zeros(N_PERTURB)
    dalpha_in = np.zeros(N_PERTURB)
    # N_space = 81
    # Mb, N_time = pick_Mb_Ntime(N_space, T_MAX)
    # N_interface = pick_Ninterface(N_space, T_MAX)
    # print(N_space, N_interface, Mb, N_time)
    perturb_residual_history = []
    perturb_l2_out_history = []
    perturb_l2_in_history = []
    # ext_pts, int_pts, if_pts, bd_pts, ext_init_pts, int_init_pts = generate_points_time_space(N_space, N_interface, Mb, 11)
    for it in range(5):  # 迭代步数可调整
        ext_pts, int_pts, if_pts, bd_pts, ext_init_pts, int_init_pts = generate_points_time_space_mc(97648, 152353, 1600, 1037, 5000)
        R, J = compute_perturb_residual_and_jacobian_sin(
            dalpha_out, dalpha_in,
            alpha_out, alpha_inner,
            ext_pts, int_pts, if_pts, bd_pts,
            ext_init_pts, int_init_pts
        )
        res_norm = np.linalg.norm(R) / np.sqrt(len(R))
        delta, _, _, _ = lstsq(J, -R, cond=1e-14)
        dalpha_out += delta[:N_PERTURB]
        dalpha_in += delta[N_PERTURB:]
        # 误差评估
        rel_out, rel_in = compute_relative_error_perturbed(
            alpha_out, alpha_inner,  # 主解权重
            dalpha_out, dalpha_in,  # 摄动权重
            EPS,  # 摄动幅度
            w_out, b_out, w_inner, b_inner,  # 主解网络参数
            w_out1, b_out1, w_inner1, b_inner1,  # 摄动网络参数
            T_MAX,
        )
        perturb_residual_history.append(res_norm)
        perturb_l2_out_history.append(rel_out)
        perturb_l2_in_history.append(rel_in)
        print(f"[Perturb {it}] Res={res_norm:.3e} RelErr_out={rel_out:.3e} RelErr_in={rel_in:.3e}")