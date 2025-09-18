import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import time
import sympy as sp

# ========================== 制造解与右端项 f(x) 的符号构造 ==========================
# 符号变量
xs, ys = sp.symbols('x y', real=True)
u_mm_expr = xs**2 * ys**2                       # Ω^{--}: x<0, y<0
u_pm_expr = sp.exp(-xs) * xs**2 * ys**2         # Ω^{+-}: x>=0,y<0
u_mp_expr = sp.exp(-ys) * xs**2 * ys**2         # Ω^{-+}: x<0,y>=0
u_pp_expr = sp.exp(-(xs+ys)) * xs**2 * ys**2    # Ω^{++}: x>=0,y>=0

u_sym = sp.symbols('u_sym', real=True)
beta_mm_u = 1 + u_sym
beta_pm_u = 1 + sp.Rational(1,2)*u_sym**2
beta_mp_u = 1 + sp.Rational(1,4)*u_sym**2
beta_pp_u = 1 + sp.Rational(1,10)*u_sym**3

dbeta_mm_u = sp.diff(beta_mm_u, u_sym)
dbeta_pm_u = sp.diff(beta_pm_u, u_sym)
dbeta_mp_u = sp.diff(beta_mp_u, u_sym)
dbeta_pp_u = sp.diff(beta_pp_u, u_sym)

def build_f(u_expr, beta_u_expr, dbeta_u_expr):
    # f = -( β Δu + β'(u) |∇u|^2 )，以精确解u代入
    ux = sp.diff(u_expr, xs)
    uy = sp.diff(u_expr, ys)
    lap = sp.diff(ux, xs) + sp.diff(uy, ys)
    grad2 = ux**2 + uy**2
    beta_xy = beta_u_expr.subs({u_sym: u_expr})
    dbeta_xy = dbeta_u_expr.subs({u_sym: u_expr})
    f_xy = -(beta_xy*lap + dbeta_xy*grad2)
    return sp.simplify(f_xy)

f_mm_expr = build_f(u_mm_expr, beta_mm_u, dbeta_mm_u)
f_pm_expr = build_f(u_pm_expr, beta_pm_u, dbeta_pm_u)
f_mp_expr = build_f(u_mp_expr, beta_mp_u, dbeta_mp_u)
f_pp_expr = build_f(u_pp_expr, beta_pp_u, dbeta_pp_u)

# lambdify为numpy函数
u_mm_fun = sp.lambdify((xs, ys), u_mm_expr, 'numpy')
u_pm_fun = sp.lambdify((xs, ys), u_pm_expr, 'numpy')
u_mp_fun = sp.lambdify((xs, ys), u_mp_expr, 'numpy')
u_pp_fun = sp.lambdify((xs, ys), u_pp_expr, 'numpy')

f_mm_fun = sp.lambdify((xs, ys), f_mm_expr, 'numpy')
f_pm_fun = sp.lambdify((xs, ys), f_pm_expr, 'numpy')
f_mp_fun = sp.lambdify((xs, ys), f_mp_expr, 'numpy')
f_pp_fun = sp.lambdify((xs, ys), f_pp_expr, 'numpy')

# ========================== 精确解与β的分片评估函数 ==========================
def exact_u_piece(x, y):
    # 返回精确解u(x,y)
    x = np.asarray(x); y = np.asarray(y)
    out = np.zeros(np.broadcast(x, y).shape, dtype=float)
    mm = (x < 0) & (y < 0)
    pm = (x >= 0) & (y < 0)
    mp = (x < 0) & (y >= 0)
    pp = (x >= 0) & (y >= 0)
    if np.any(mm): out[mm] = u_mm_fun(x[mm], y[mm])
    if np.any(pm): out[pm] = u_pm_fun(x[pm], y[pm])
    if np.any(mp): out[mp] = u_mp_fun(x[mp], y[mp])
    if np.any(pp): out[pp] = u_pp_fun(x[pp], y[pp])
    return out

def f_piece(x, y):
    # 返回制造项f(x,y)
    x = np.asarray(x); y = np.asarray(y)
    out = np.zeros(np.broadcast(x, y).shape, dtype=float)
    mm = (x < 0) & (y < 0)
    pm = (x >= 0) & (y < 0)
    mp = (x < 0) & (y >= 0)
    pp = (x >= 0) & (y >= 0)
    if np.any(mm): out[mm] = f_mm_fun(x[mm], y[mm])
    if np.any(pm): out[pm] = f_pm_fun(x[pm], y[pm])
    if np.any(mp): out[mp] = f_mp_fun(x[mp], y[mp])
    if np.any(pp): out[pp] = f_pp_fun(x[pp], y[pp])
    return out

# β及其导数（按子域）工具
def beta_and_derivs(region, u):
    # region ∈ {'mm','pm','mp','pp'}
    if region == 'mm':
        beta = 1.0 + u
        beta_u = 1.0
        beta_uu = 0.0
        beta_uuu = 0.0
    elif region == 'pm':
        beta = 1.0 + 0.5 * u*u
        beta_u = u
        beta_uu = 1.0
        beta_uuu = 0.0
    elif region == 'mp':
        beta = 1.0 + 0.25 * u*u
        beta_u = 0.5 * u
        beta_uu = 0.5
        beta_uuu = 0.0
    else:  # 'pp'
        beta = 1.0 + 0.1 * (u**3)
        beta_u = 0.3 * (u**2)
        beta_uu = 0.6 * u
        beta_uuu = 0.6
    return beta, beta_u, beta_uu, beta_uuu

# 精确解各子域
def u_exact_mm(x,y): return u_mm_fun(x,y)
def u_exact_pm(x,y): return u_pm_fun(x,y)
def u_exact_mp(x,y): return u_mp_fun(x,y)
def u_exact_pp(x,y): return u_pp_fun(x,y)

# ========================== 采样点生成（四子域+两条界面+边界） ==========================
def generate_points_4(Nxy, N_interface, Mb):
    # Nxy: 每条轴方向的半区间采样数（内部PDE点用）
    # N_interface: 每条接口上的采样点数
    # Mb: 边界总采样数（近似均匀分配到四条边，再按象限划分）
    x = np.linspace(-1, 1, 2*Nxy)
    y = np.linspace(-1, 1, 2*Nxy)
    xx, yy = np.meshgrid(x, y, indexing='xy')

    # 内部点（严格避开界面）
    mask_mm = (xx < 0) & (yy < 0)
    mask_pm = (xx >= 0) & (yy < 0)
    mask_mp = (xx < 0) & (yy >= 0)
    mask_pp = (xx >= 0) & (yy >= 0)

    # 去掉界面上的点
    mask_mm &= (xx != 0) & (yy != 0)
    mask_pm &= (xx != 0) & (yy != 0)
    mask_mp &= (xx != 0) & (yy != 0)
    mask_pp &= (xx != 0) & (yy != 0)

    X_mm = np.column_stack([xx[mask_mm], yy[mask_mm]])
    X_pm = np.column_stack([xx[mask_pm], yy[mask_pm]])
    X_mp = np.column_stack([xx[mask_mp], yy[mask_mp]])
    X_pp = np.column_stack([xx[mask_pp], yy[mask_pp]])

    # 两条界面上的点
    gamma_x = np.column_stack([np.zeros(N_interface), np.linspace(-1, 1, N_interface)])  # x=0
    gamma_y = np.column_stack([np.linspace(-1, 1, N_interface), np.zeros(N_interface)])  # y=0

    # 边界点：四条边
    m = Mb // 4
    left = np.column_stack([ -np.ones(m), np.linspace(-1, 1, m) ])
    right= np.column_stack([  np.ones(m), np.linspace(-1, 1, m) ])
    bottom=np.column_stack([ np.linspace(-1, 1, m), -np.ones(m) ])
    top   =np.column_stack([ np.linspace(-1, 1, m),  np.ones(m) ])

    # 按子域划分边界点
    # 左边界x=-1：y<0 -> mm，y>=0 -> mp
    bc_mm = left[left[:,1] < 0]
    bc_mp = left[left[:,1] >= 0]
    # 右边界x=+1：y<0 -> pm，y>=0 -> pp
    bc_pm = right[right[:,1] < 0]
    bc_pp = right[right[:,1] >= 0]
    # 底边界y=-1：x<0 -> mm，x>=0 -> pm
    bc_mm = np.vstack([bc_mm, bottom[bottom[:,0] < 0]])
    bc_pm = np.vstack([bc_pm, bottom[bottom[:,0] >= 0]])
    # 顶边界y=+1：x<0 -> mp，x>=0 -> pp
    bc_mp = np.vstack([bc_mp, top[top[:,0] < 0]])
    bc_pp = np.vstack([bc_pp, top[top[:,0] >= 0]])

    return X_mm, X_pm, X_mp, X_pp, gamma_x, gamma_y, bc_mm, bc_pm, bc_mp, bc_pp

# ========================== ELM参数与工具 ==========================
np.random.seed(42)
N = 100                      # 每个子域主网络隐层数(tanh)
N_perturb = 400              # 每个子域摄动网络隐层数(sin)
gamma_bc = 500.0
gamma_interface = 1000.0

# 四个子域的tanh-ELM随机权重与偏置
def init_elm_tanh(N):
    w = 1.0 * np.random.randn(N, 2)
    b = 0.1 * np.random.randn(N)
    return w, b

w_mm, b_mm = init_elm_tanh(N)
w_pm, b_pm = init_elm_tanh(N)
w_mp, b_mp = init_elm_tanh(N)
w_pp, b_pp = init_elm_tanh(N)

# 四个子域的sin-ELM随机权重与偏置（摄动网络）
def init_elm_sin(Np):
    w = 5 * np.pi * np.random.randn(Np, 2)
    b = 1.0 * np.random.randn(Np)
    return w, b

w_mm1, b_mm1 = init_elm_sin(N_perturb)
w_pm1, b_pm1 = init_elm_sin(N_perturb)
w_mp1, b_mp1 = init_elm_sin(N_perturb)
w_pp1, b_pp1 = init_elm_sin(N_perturb)

# 激活及其导数（tanh）
def tanh_feats(w, b, x, y):
    z = w[:,0] * x + w[:,1] * y + b
    t = np.tanh(z)
    tp = 1 - t**2
    tpp = -2 * t * tp
    return z, t, tp, tpp

# 激活及其导数（sin）
def sin_feats(w, b, x, y):
    z = w[:,0] * x + w[:,1] * y + b
    s = np.sin(z)
    c = np.cos(z)
    # Laplacian of sin(w·x+b) = -|w|^2 sin(z)
    return z, s, c

# 计算u, grad, laplace（tanh-ELM）
def eval_tanh_elm(alpha, w, b, x, y):
    _, t, tp, tpp = tanh_feats(w, b, x, y)
    u = np.dot(alpha, t)
    ux = np.dot(alpha, w[:,0] * tp)
    uy = np.dot(alpha, w[:,1] * tp)
    lap = np.dot(alpha, (w[:,0]**2 + w[:,1]**2) * tpp)
    return u, ux, uy, lap, t, tp, tpp

# 计算u, grad, laplace（sin-ELM）
def eval_sin_elm(alpha, w, b, x, y):
    _, s, c = sin_feats(w, b, x, y)
    u = np.dot(alpha, s)
    ux = np.dot(alpha, w[:,0] * c)
    uy = np.dot(alpha, w[:,1] * c)
    lap = np.dot(alpha, (w[:,0]**2 + w[:,1]**2) * (-s))
    return u, ux, uy, lap, s, c

# ========================== 主问题：残差与雅可比（四子域） ==========================
def compute_residual_and_jacobian_4(alpha_mm, alpha_pm, alpha_mp, alpha_pp,
                                    X_mm, X_pm, X_mp, X_pp, gamma_x, gamma_y,
                                    bc_mm, bc_pm, bc_mp, bc_pp):
    # 总未知维数：4*N
    # 行分块：四子域PDE + 两条界面(各自的连续与通量) + 四子域边界
    M_mm = len(X_mm); M_pm = len(X_pm); M_mp = len(X_mp); M_pp = len(X_pp)
    M_gx = len(gamma_x); M_gy = len(gamma_y)
    M_bc_mm = len(bc_mm); M_bc_pm = len(bc_pm); M_bc_mp = len(bc_mp); M_bc_pp = len(bc_pp)

    # 两条界面均有两类条件（连续与通量）
    M_total = (M_mm + M_pm + M_mp + M_pp) + 2*M_gx + 2*M_gy + (M_bc_mm + M_bc_pm + M_bc_mp + M_bc_pp)
    R = np.zeros(M_total)
    J = np.zeros((M_total, 4*N))

    row = 0

    # 子域PDE残差 f(x) 只依赖(x,y)，牛顿雅可比里 f_u=0
    # Ω^{--}
    for j in range(M_mm):
        x,y = X_mm[j]
        u, ux, uy, lap, t, tp, tpp = eval_tanh_elm(alpha_mm, w_mm, b_mm, x, y)
        beta, beta_u, beta_uu, _ = beta_and_derivs('mm', u)
        grad2 = ux**2 + uy**2
        fxy = f_mm_fun(x, y)
        R[row] = beta * lap + beta_u * grad2 + fxy
        J[row, 0:N] = (beta_u * lap) * t + beta * (w_mm[:,0]**2 + w_mm[:,1]**2) * tpp + 2 * beta_u * (ux * w_mm[:,0] * tp + uy * w_mm[:,1] * tp) + beta_uu * grad2 * t
        row += 1

    # Ω^{+-}
    for j in range(M_pm):
        x,y = X_pm[j]
        u, ux, uy, lap, t, tp, tpp = eval_tanh_elm(alpha_pm, w_pm, b_pm, x, y)
        beta, beta_u, beta_uu, _ = beta_and_derivs('pm', u)
        grad2 = ux**2 + uy**2
        fxy = f_pm_fun(x, y)
        R[row] = beta * lap + beta_u * grad2 + fxy
        J[row, N:2*N] = (beta_u * lap) * t + beta * (w_pm[:,0]**2 + w_pm[:,1]**2) * tpp + 2 * beta_u * (ux * w_pm[:,0] * tp + uy * w_pm[:,1] * tp) + beta_uu * grad2 * t
        row += 1

    # Ω^{-+}
    for j in range(M_mp):
        x,y = X_mp[j]
        u, ux, uy, lap, t, tp, tpp = eval_tanh_elm(alpha_mp, w_mp, b_mp, x, y)
        beta, beta_u, beta_uu, _ = beta_and_derivs('mp', u)
        grad2 = ux**2 + uy**2
        fxy = f_mp_fun(x, y)
        R[row] = beta * lap + beta_u * grad2 + fxy
        J[row, 2*N:3*N] = (beta_u * lap) * t + beta * (w_mp[:,0]**2 + w_mp[:,1]**2) * tpp + 2 * beta_u * (ux * w_mp[:,0] * tp + uy * w_mp[:,1] * tp) + beta_uu * grad2 * t
        row += 1

    # Ω^{++}
    for j in range(M_pp):
        x,y = X_pp[j]
        u, ux, uy, lap, t, tp, tpp = eval_tanh_elm(alpha_pp, w_pp, b_pp, x, y)
        beta, beta_u, beta_uu, _ = beta_and_derivs('pp', u)
        grad2 = ux**2 + uy**2
        fxy = f_pp_fun(x, y)
        R[row] = beta * lap + beta_u * grad2 + fxy
        J[row, 3*N:4*N] = (beta_u * lap) * t + beta * (w_pp[:,0]**2 + w_pp[:,1]**2) * tpp + 2 * beta_u * (ux * w_pp[:,0] * tp + uy * w_pp[:,1] * tp) + beta_uu * grad2 * t
        row += 1

    # 界面 x=0：y<0时 mm↔pm，y>=0时 mp↔pp
    # 连续性[u]=0
    for j in range(M_gx):
        x, y = gamma_x[j]  # x=0
        # 下半
        if y < 0:
            uL, uxL, uyL, lapL, tL, tpL, _ = eval_tanh_elm(alpha_mm, w_mm, b_mm, x, y)
            uR, uxR, uyR, lapR, tR, tpR, _ = eval_tanh_elm(alpha_pm, w_pm, b_pm, x, y)
            R[row] = (uR - uL) * gamma_interface
            J[row, N:2*N] = tR * gamma_interface
            J[row, 0:N]  = -tL * gamma_interface
        else:
            uL, uxL, uyL, lapL, tL, tpL, _ = eval_tanh_elm(alpha_mp, w_mp, b_mp, x, y)
            uR, uxR, uyR, lapR, tR, tpR, _ = eval_tanh_elm(alpha_pp, w_pp, b_pp, x, y)
            R[row] = (uR - uL) * gamma_interface
            J[row, 3*N:4*N] = tR * gamma_interface
            J[row, 2*N:3*N] = -tL * gamma_interface
        row += 1
    # 通量[β ∂u/∂n]=0，法向为x方向
    for j in range(M_gx):
        x, y = gamma_x[j]  # x=0
        if y < 0:
            uL, uxL, uyL, _, tL, tpL, _ = eval_tanh_elm(alpha_mm, w_mm, b_mm, x, y)
            uR, uxR, uyR, _, tR, tpR, _ = eval_tanh_elm(alpha_pm, w_pm, b_pm, x, y)
            betaL, betaL_u, _, _ = beta_and_derivs('mm', uL)
            betaR, betaR_u, _, _ = beta_and_derivs('pm', uR)
            R[row] = (betaR * uxR - betaL * uxL) * gamma_interface
            J[row, N:2*N] = (betaR_u * uxR * tR + betaR * w_pm[:,0] * tpR) * gamma_interface
            J[row, 0:N]  = -(betaL_u * uxL * tL + betaL * w_mm[:,0] * tpL) * gamma_interface
        else:
            uL, uxL, uyL, _, tL, tpL, _ = eval_tanh_elm(alpha_mp, w_mp, b_mp, x, y)
            uR, uxR, uyR, _, tR, tpR, _ = eval_tanh_elm(alpha_pp, w_pp, b_pp, x, y)
            betaL, betaL_u, _, _ = beta_and_derivs('mp', uL)
            betaR, betaR_u, _, _ = beta_and_derivs('pp', uR)
            R[row] = (betaR * uxR - betaL * uxL) * gamma_interface
            J[row, 3*N:4*N] = (betaR_u * uxR * tR + betaR * w_pp[:,0] * tpR) * gamma_interface
            J[row, 2*N:3*N] = -(betaL_u * uxL * tL + betaL * w_mp[:,0] * tpL) * gamma_interface
        row += 1

    # 界面 y=0：x<0时 mm↔mp，x>=0时 pm↔pp
    # 连续性[u]=0
    for j in range(M_gy):
        x, y = gamma_y[j]  # y=0
        if x < 0:
            uB, uxB, uyB, _, tB, tpB, _ = eval_tanh_elm(alpha_mm, w_mm, b_mm, x, y)
            uT, uxT, uyT, _, tT, tpT, _ = eval_tanh_elm(alpha_mp, w_mp, b_mp, x, y)
            R[row] = (uT - uB) * gamma_interface
            J[row, 2*N:3*N] = tT * gamma_interface
            J[row, 0:N]     = -tB * gamma_interface
        else:
            uB, uxB, uyB, _, tB, tpB, _ = eval_tanh_elm(alpha_pm, w_pm, b_pm, x, y)
            uT, uxT, uyT, _, tT, tpT, _ = eval_tanh_elm(alpha_pp, w_pp, b_pp, x, y)
            R[row] = (uT - uB) * gamma_interface
            J[row, 3*N:4*N] = tT * gamma_interface
            J[row, N:2*N]   = -tB * gamma_interface
        row += 1
    # 通量[β ∂u/∂n]=0，法向为y方向
    for j in range(M_gy):
        x, y = gamma_y[j]  # y=0
        if x < 0:
            uB, uxB, uyB, _, tB, tpB, _ = eval_tanh_elm(alpha_mm, w_mm, b_mm, x, y)
            uT, uxT, uyT, _, tT, tpT, _ = eval_tanh_elm(alpha_mp, w_mp, b_mp, x, y)
            betaB, betaB_u, _, _ = beta_and_derivs('mm', uB)
            betaT, betaT_u, _, _ = beta_and_derivs('mp', uT)
            R[row] = (betaT * uyT - betaB * uyB) * gamma_interface
            J[row, 2*N:3*N] = (betaT_u * uyT * tT + betaT * w_mp[:,1] * tpT) * gamma_interface
            J[row, 0:N]     = -(betaB_u * uyB * tB + betaB * w_mm[:,1] * tpB) * gamma_interface
        else:
            uB, uxB, uyB, _, tB, tpB, _ = eval_tanh_elm(alpha_pm, w_pm, b_pm, x, y)
            uT, uxT, uyT, _, tT, tpT, _ = eval_tanh_elm(alpha_pp, w_pp, b_pp, x, y)
            betaB, betaB_u, _, _ = beta_and_derivs('pm', uB)
            betaT, betaT_u, _, _ = beta_and_derivs('pp', uT)
            R[row] = (betaT * uyT - betaB * uyB) * gamma_interface
            J[row, 3*N:4*N] = (betaT_u * uyT * tT + betaT * w_pp[:,1] * tpT) * gamma_interface
            J[row, N:2*N]   = -(betaB_u * uyB * tB + betaB * w_pm[:,1] * tpB) * gamma_interface
        row += 1

    # 边界Dirichlet：各子域对应边界上的点，u=精确解
    for j in range(M_bc_mm):
        x,y = bc_mm[j]
        u, ux, uy, lap, t, tp, tpp = eval_tanh_elm(alpha_mm, w_mm, b_mm, x, y)
        R[row] = (u - u_mm_fun(x,y)) * gamma_bc
        J[row, 0:N] = t * gamma_bc
        row += 1
    for j in range(M_bc_pm):
        x,y = bc_pm[j]
        u, ux, uy, lap, t, tp, tpp = eval_tanh_elm(alpha_pm, w_pm, b_pm, x, y)
        R[row] = (u - u_pm_fun(x,y)) * gamma_bc
        J[row, N:2*N] = t * gamma_bc
        row += 1
    for j in range(M_bc_mp):
        x,y = bc_mp[j]
        u, ux, uy, lap, t, tp, tpp = eval_tanh_elm(alpha_mp, w_mp, b_mp, x, y)
        R[row] = (u - u_mp_fun(x,y)) * gamma_bc
        J[row, 2*N:3*N] = t * gamma_bc
        row += 1
    for j in range(M_bc_pp):
        x,y = bc_pp[j]
        u, ux, uy, lap, t, tp, tpp = eval_tanh_elm(alpha_pp, w_pp, b_pp, x, y)
        R[row] = (u - u_pp_fun(x,y)) * gamma_bc
        J[row, 3*N:4*N] = t * gamma_bc
        row += 1

    return R, J

# ========================== 摄动阶段：残差与雅可比（四子域） ==========================
def compute_perturb_residual_4(alpha_mm_p, alpha_pm_p, alpha_mp_p, alpha_pp_p,
                               alpha_mm, alpha_pm, alpha_mp, alpha_pp,
                               X_mm, X_pm, X_mp, X_pp, gamma_x, gamma_y,
                               bc_mm, bc_pm, bc_mp, bc_pp, epsilon, gamma=100.0):
    # 该函数将你原有两子域摄动函数推广到四子域与两条界面
    # 行结构与主问题相同，但每条界面条件各自分为连续和通量两类
    # 未知为4*N_perturb
    M_mm = len(X_mm); M_pm = len(X_pm); M_mp = len(X_mp); M_pp = len(X_pp)
    M_gx = len(gamma_x); M_gy = len(gamma_y)
    M_bc_mm = len(bc_mm); M_bc_pm = len(bc_pm); M_bc_mp = len(bc_mp); M_bc_pp = len(bc_pp)
    M_total = (M_mm + M_pm + M_mp + M_pp) + 2*M_gx + 2*M_gy + (M_bc_mm + M_bc_pm + M_bc_mp + M_bc_pp)

    R = np.zeros(M_total)
    J = np.zeros((M_total, 4*N_perturb))

    row = 0

    # 工具：一处计算主解u0与摄动u1及其导数
    def main_fields(region, x, y):
        if region == 'mm':
            u0, ux0, uy0, lap0, t0, tp0, tpp0 = eval_tanh_elm(alpha_mm, w_mm, b_mm, x, y)
        elif region == 'pm':
            u0, ux0, uy0, lap0, t0, tp0, tpp0 = eval_tanh_elm(alpha_pm, w_pm, b_pm, x, y)
        elif region == 'mp':
            u0, ux0, uy0, lap0, t0, tp0, tpp0 = eval_tanh_elm(alpha_mp, w_mp, b_mp, x, y)
        else:
            u0, ux0, uy0, lap0, t0, tp0, tpp0 = eval_tanh_elm(alpha_pp, w_pp, b_pp, x, y)
        return u0, ux0, uy0, lap0

    def pert_fields(region, x, y):
        if region == 'mm':
            u1, ux1, uy1, lap1, s, c = eval_sin_elm(alpha_mm_p, w_mm1, b_mm1, x, y)
            w1, sgn = w_mm1, 0
            col_slice = slice(0, N_perturb)
        elif region == 'pm':
            u1, ux1, uy1, lap1, s, c = eval_sin_elm(alpha_pm_p, w_pm1, b_pm1, x, y)
            w1, sgn = w_pm1, 1
            col_slice = slice(N_perturb, 2*N_perturb)
        elif region == 'mp':
            u1, ux1, uy1, lap1, s, c = eval_sin_elm(alpha_mp_p, w_mp1, b_mp1, x, y)
            w1, sgn = w_mp1, 2
            col_slice = slice(2*N_perturb, 3*N_perturb)
        else:
            u1, ux1, uy1, lap1, s, c = eval_sin_elm(alpha_pp_p, w_pp1, b_pp1, x, y)
            w1, sgn = w_pp1, 3
            col_slice = slice(3*N_perturb, 4*N_perturb)
        return u1, ux1, uy1, lap1, s, c, w1, col_slice

    # 每个子域：线性化残差（结构参考你原函数），f_u=0，故对应项省略
    # Ω^{--}
    for j in range(M_mm):
        x,y = X_mm[j]
        u0, ux0, uy0, lap0 = main_fields('mm', x, y)
        u1, ux1, uy1, lap1, s, c, w1, col = pert_fields('mm', x, y)
        beta, beta_u, beta_uu, beta_uuu = beta_and_derivs('mm', u0)
        grad2 = ux0**2 + uy0**2
        # 主残差/epsilon + 一阶项 + 可选二阶项
        R0 = (beta_u * grad2 + beta * lap0 + f_mm_fun(x,y))/epsilon \
             + 2*beta_u*(ux0*ux1 + uy0*uy1) + beta_uu*grad2*u1 + beta_u*u1*lap0 + beta*lap1 \
             + epsilon*( beta_u*(ux1**2 + uy1**2) + 0.5*u1**2*beta_uu*lap0 + 0.5*u1**2*0.0*grad2 + u1*beta_u*lap1 )
        R[row] = R0
        J[row, col] = 2*beta_u*(ux0*w1[:,0]*c + uy0*w1[:,1]*c) + beta_uu*grad2*s + beta_u*lap0*s + beta*(w1[:,0]**2 + w1[:,1]**2)*(-s) \
                      + epsilon*( 2*beta_u*(ux1*w1[:,0]*c + uy1*w1[:,1]*c) + beta_u*lap1*s + beta_u*u1*(w1[:,0]**2 + w1[:,1]**2)*(-s) )
        row += 1

    # Ω^{+-}
    for j in range(M_pm):
        x,y = X_pm[j]
        u0, ux0, uy0, lap0 = main_fields('pm', x, y)
        u1, ux1, uy1, lap1, s, c, w1, col = pert_fields('pm', x, y)
        beta, beta_u, beta_uu, beta_uuu = beta_and_derivs('pm', u0)
        grad2 = ux0**2 + uy0**2
        R0 = (beta_u * grad2 + beta * lap0 + f_pm_fun(x,y))/epsilon \
             + 2*beta_u*(ux0*ux1 + uy0*uy1) + beta_uu*grad2*u1 + beta_u*u1*lap0 + beta*lap1 \
             + epsilon*( beta_u*(ux1**2 + uy1**2) + 0.5*u1**2*beta_uu*lap0 + 0.5*u1**2*beta_uuu*grad2 + u1*beta_u*lap1 )
        R[row] = R0
        J[row, col] = 2*beta_u*(ux0*w1[:,0]*c + uy0*w1[:,1]*c) + beta_uu*grad2*s + beta_u*lap0*s + beta*(w1[:,0]**2 + w1[:,1]**2)*(-s) \
                      + epsilon*( 2*beta_u*(ux1*w1[:,0]*c + uy1*w1[:,1]*c) + u1*beta_uuu*grad2*s + beta_u*lap1*s + beta_u*u1*(w1[:,0]**2 + w1[:,1]**2)*(-s) )
        row += 1

    # Ω^{-+}
    for j in range(M_mp):
        x,y = X_mp[j]
        u0, ux0, uy0, lap0 = main_fields('mp', x, y)
        u1, ux1, uy1, lap1, s, c, w1, col = pert_fields('mp', x, y)
        beta, beta_u, beta_uu, beta_uuu = beta_and_derivs('mp', u0)
        grad2 = ux0**2 + uy0**2
        R0 = (beta_u * grad2 + beta * lap0 + f_mp_fun(x,y))/epsilon \
             + 2*beta_u*(ux0*ux1 + uy0*uy1) + beta_uu*grad2*u1 + beta_u*u1*lap0 + beta*lap1 \
             + epsilon*( beta_u*(ux1**2 + uy1**2) + 0.5*u1**2*beta_uu*lap0 + 0.5*u1**2*beta_uuu*grad2 + u1*beta_u*lap1 )
        R[row] = R0
        J[row, col] = 2*beta_u*(ux0*w1[:,0]*c + uy0*w1[:,1]*c) + beta_uu*grad2*s + beta_u*lap0*s + beta*(w1[:,0]**2 + w1[:,1]**2)*(-s) \
                      + epsilon*( 2*beta_u*(ux1*w1[:,0]*c + uy1*w1[:,1]*c) + u1*beta_uuu*grad2*s + beta_u*lap1*s + beta_u*u1*(w1[:,0]**2 + w1[:,1]**2)*(-s) )
        row += 1

    # Ω^{++}
    for j in range(M_pp):
        x,y = X_pp[j]
        u0, ux0, uy0, lap0 = main_fields('pp', x, y)
        u1, ux1, uy1, lap1, s, c, w1, col = pert_fields('pp', x, y)
        beta, beta_u, beta_uu, beta_uuu = beta_and_derivs('pp', u0)
        grad2 = ux0**2 + uy0**2
        R0 = (beta_u * grad2 + beta * lap0 + f_pp_fun(x,y))/epsilon \
             + 2*beta_u*(ux0*ux1 + uy0*uy1) + beta_uu*grad2*u1 + beta_u*u1*lap0 + beta*lap1 \
             + epsilon*( beta_u*(ux1**2 + uy1**2) + 0.5*u1**2*beta_uu*lap0 + 0.5*u1**2*beta_uuu*grad2 + u1*beta_u*lap1 )
        R[row] = R0
        J[row, col] = 2*beta_u*(ux0*w1[:,0]*c + uy0*w1[:,1]*c) + beta_uu*grad2*s + beta_u*lap0*s + beta*(w1[:,0]**2 + w1[:,1]**2)*(-s) \
                      + epsilon*( 2*beta_u*(ux1*w1[:,0]*c + uy1*w1[:,1]*c) + u1*beta_uuu*grad2*s + beta_u*lap1*s + beta_u*u1*(w1[:,0]**2 + w1[:,1]**2)*(-s) )
        row += 1

    # 界面 x=0：连续 + 通量(法向x)
    for j in range(M_gx):
        x,y = gamma_x[j]
        if y < 0:
            # mm ↔ pm
            uL0, uxL0, uyL0, _ = main_fields('mm', x, y)
            uR0, uxR0, uyR0, _ = main_fields('pm', x, y)
            uL1, uxL1, uyL1, _, sL, cL, wL, colL = pert_fields('mm', x, y)
            uR1, uxR1, uyR1, _, sR, cR, wR, colR = pert_fields('pm', x, y)
            betaL, betaL_u, betaL_uu, _ = beta_and_derivs('mm', uL0)
            betaR, betaR_u, betaR_uu, _ = beta_and_derivs('pm', uR0)
            # 连续
            R[row] = ((uR0 - uL0)/epsilon + (uR1 - uL1)) * gamma
            J[row, colR] = sR * gamma
            J[row, colL] = -sL * gamma
            row += 1
            # 通量
            R[row] = ((betaR*uxR0 - betaL*uxL0)/epsilon + (betaR_u*uR1*uxR0 + betaR*uxR1 - betaL_u*uL1*uxL0 - betaL*uxL1)) * gamma
            J[row, colR] = (betaR_u * sR * uxR0 + betaR * wR[:,0] * cR) * gamma
            J[row, colL] = -(betaL_u * sL * uxL0 + betaL * wL[:,0] * cL) * gamma
            row += 1
        else:
            # mp ↔ pp
            uL0, uxL0, uyL0, _ = main_fields('mp', x, y)
            uR0, uxR0, uyR0, _ = main_fields('pp', x, y)
            uL1, uxL1, uyL1, _, sL, cL, wL, colL = pert_fields('mp', x, y)
            uR1, uxR1, uyR1, _, sR, cR, wR, colR = pert_fields('pp', x, y)
            betaL, betaL_u, betaL_uu, _ = beta_and_derivs('mp', uL0)
            betaR, betaR_u, betaR_uu, _ = beta_and_derivs('pp', uR0)
            R[row] = ((uR0 - uL0)/epsilon + (uR1 - uL1)) * gamma
            J[row, colR] = sR * gamma
            J[row, colL] = -sL * gamma
            row += 1
            R[row] = ((betaR*uxR0 - betaL*uxL0)/epsilon + (betaR_u*uR1*uxR0 + betaR*uxR1 - betaL_u*uL1*uxL0 - betaL*uxL1)) * gamma
            J[row, colR] = (betaR_u * sR * uxR0 + betaR * wR[:,0] * cR) * gamma
            J[row, colL] = -(betaL_u * sL * uxL0 + betaL * wL[:,0] * cL) * gamma
            row += 1

    # 界面 y=0：连续 + 通量(法向y)
    for j in range(M_gy):
        x,y = gamma_y[j]
        if x < 0:
            # mm ↔ mp
            uB0, uxB0, uyB0, _ = main_fields('mm', x, y)
            uT0, uxT0, uyT0, _ = main_fields('mp', x, y)
            uB1, uxB1, uyB1, _, sB, cB, wB, colB = pert_fields('mm', x, y)
            uT1, uxT1, uyT1, _, sT, cT, wT, colT = pert_fields('mp', x, y)
            betaB, betaB_u, betaB_uu, _ = beta_and_derivs('mm', uB0)
            betaT, betaT_u, betaT_uu, _ = beta_and_derivs('mp', uT0)
            R[row] = ((uT0 - uB0)/epsilon + (uT1 - uB1)) * gamma
            J[row, colT] = sT * gamma
            J[row, colB] = -sB * gamma
            row += 1
            R[row] = ((betaT*uyT0 - betaB*uyB0)/epsilon + (betaT_u*uT1*uyT0 + betaT*uyT1 - betaB_u*uB1*uyB0 - betaB*uyB1)) * gamma
            J[row, colT] = (betaT_u * sT * uyT0 + betaT * wT[:,1] * cT) * gamma
            J[row, colB] = -(betaB_u * sB * uyB0 + betaB * wB[:,1] * cB) * gamma
            row += 1
        else:
            # pm ↔ pp
            uB0, uxB0, uyB0, _ = main_fields('pm', x, y)
            uT0, uxT0, uyT0, _ = main_fields('pp', x, y)
            uB1, uxB1, uyB1, _, sB, cB, wB, colB = pert_fields('pm', x, y)
            uT1, uxT1, uyT1, _, sT, cT, wT, colT = pert_fields('pp', x, y)
            betaB, betaB_u, betaB_uu, _ = beta_and_derivs('pm', uB0)
            betaT, betaT_u, betaT_uu, _ = beta_and_derivs('pp', uT0)
            R[row] = ((uT0 - uB0)/epsilon + (uT1 - uB1)) * gamma
            J[row, colT] = sT * gamma
            J[row, colB] = -sB * gamma
            row += 1
            R[row] = ((betaT*uyT0 - betaB*uyB0)/epsilon + (betaT_u*uT1*uyT0 + betaT*uyT1 - betaB_u*uB1*uyB0 - betaB*uyB1)) * gamma
            J[row, colT] = (betaT_u * sT * uyT0 + betaT * wT[:,1] * cT) * gamma
            J[row, colB] = -(betaB_u * sB * uyB0 + betaB * wB[:,1] * cB) * gamma
            row += 1

            # 边界条件： (u0 + ε u1 - u_exact)/ε = 0
    for j in range(M_bc_mm):
        x, y = bc_mm[j]
        # u0, _, _, _ = eval_tanh_elm(alpha_mm, w_mm, b_mm, x, y)   # 错误：eval_tanh_elm返回7个量
        u0, _, _, _ = main_fields('mm', x, y)  # 修复：使用main_fields
        u1, _, _, _, s, c, w1, col = pert_fields('mm', x, y)
        R[row] = (u0 + epsilon * u1 - u_mm_fun(x, y)) / epsilon * gamma
        J[row, col] = s * gamma
        row += 1

    for j in range(M_bc_pm):
        x, y = bc_pm[j]
        # u0, _, _, _ = eval_tanh_elm(alpha_pm, w_pm, b_pm, x, y)
        u0, _, _, _ = main_fields('pm', x, y)
        u1, _, _, _, s, c, w1, col = pert_fields('pm', x, y)
        R[row] = (u0 + epsilon * u1 - u_pm_fun(x, y)) / epsilon * gamma
        J[row, col] = s * gamma
        row += 1

    for j in range(M_bc_mp):
        x, y = bc_mp[j]
        # u0, _, _, _ = eval_tanh_elm(alpha_mp, w_mp, b_mp, x, y)
        u0, _, _, _ = main_fields('mp', x, y)
        u1, _, _, _, s, c, w1, col = pert_fields('mp', x, y)
        R[row] = (u0 + epsilon * u1 - u_mp_fun(x, y)) / epsilon * gamma
        J[row, col] = s * gamma
        row += 1

    for j in range(M_bc_pp):
        x, y = bc_pp[j]
        # u0, _, _, _ = eval_tanh_elm(alpha_pp, w_pp, b_pp, x, y)
        u0, _, _, _ = main_fields('pp', x, y)
        u1, _, _, _, s, c, w1, col = pert_fields('pp', x, y)
        R[row] = (u0 + epsilon * u1 - u_pp_fun(x, y)) / epsilon * gamma
        J[row, col] = s * gamma
        row += 1

    return R, J

# ========================== 主循环（牛顿-ELM）与摄动阶段 ==========================
# 初值
alpha_mm = np.zeros(N)
alpha_pm = np.zeros(N)
alpha_mp = np.zeros(N)
alpha_pp = np.zeros(N)

tol = 1e-10
max_iter = 10
delta_threshold = 1e-6

# 采样点
X_mm, X_pm, X_mp, X_pp, gamma_x, gamma_y, bc_mm, bc_pm, bc_mp, bc_pp = generate_points_4(Nxy=64, N_interface=161, Mb=2400)

main_residual_history = []
prev_residual = float('inf')
start_time = time.time()

for k in range(max_iter):
    R, J = compute_residual_and_jacobian_4(alpha_mm, alpha_pm, alpha_mp, alpha_pp,
                                           X_mm, X_pm, X_mp, X_pp, gamma_x, gamma_y,
                                           bc_mm, bc_pm, bc_mp, bc_pp)
    M_total = len(R)
    residual_norm = np.linalg.norm(R)/np.sqrt(M_total)
    main_residual_history.append(residual_norm)
    print(f"Main Iter {k}: Residual norm = {residual_norm:.4e}")
    if abs(residual_norm - prev_residual) < delta_threshold:
        print(f"Main iteration stopped: residual diff < {delta_threshold}")
        break
    prev_residual = residual_norm
    if residual_norm < tol:
        break
    # 最小二乘更新
    delta_alpha, _, _, _ = lstsq(J, -R, cond=1e-12)
    alpha_mm += delta_alpha[0:N]
    alpha_pm += delta_alpha[N:2*N]
    alpha_mp += delta_alpha[2*N:3*N]
    alpha_pp += delta_alpha[3*N:4*N]

print("Main M_total =", M_total)

# 摄动阶段
epsilon = 1e-4
alpha_mm_p = np.zeros(N_perturb)
alpha_pm_p = np.zeros(N_perturb)
alpha_mp_p = np.zeros(N_perturb)
alpha_pp_p = np.zeros(N_perturb)

# 更密集的采样
X_mm_p, X_pm_p, X_mp_p, X_pp_p, gamma_x_p, gamma_y_p, bc_mm_p, bc_pm_p, bc_mp_p, bc_pp_p = generate_points_4(Nxy=80, N_interface=201, Mb=4000)

max_perturb_iter = 10
perturb_tol = 1e-10
perturb_residual_history = []
prev_perturb_residual = float('inf')

for k in range(max_perturb_iter):
    R_p, J_p = compute_perturb_residual_4(alpha_mm_p, alpha_pm_p, alpha_mp_p, alpha_pp_p,
                                          alpha_mm, alpha_pm, alpha_mp, alpha_pp,
                                          X_mm_p, X_pm_p, X_mp_p, X_pp_p, gamma_x_p, gamma_y_p,
                                          bc_mm_p, bc_pm_p, bc_mp_p, bc_pp_p, epsilon, gamma=100.0)
    M_total_p = len(R_p)
    res_norm = np.linalg.norm(R_p)/np.sqrt(M_total_p)
    print(f"Perturb Iter {k}: Residual = {res_norm:.3e}")
    perturb_residual_history.append(res_norm)
    if abs(res_norm - prev_perturb_residual) < delta_threshold:
        print(f"Perturb iteration stopped: residual diff < {delta_threshold}")
        break
    prev_perturb_residual = res_norm
    if res_norm < perturb_tol:
        break
    delta_p, _, _, _ = lstsq(J_p, -R_p, cond=None)
    alpha_mm_p += delta_p[0:N_perturb]
    alpha_pm_p += delta_p[N_perturb:2*N_perturb]
    alpha_mp_p += delta_p[2*N_perturb:3*N_perturb]
    alpha_pp_p += delta_p[3*N_perturb:4*N_perturb]

end_time = time.time()
print(f"耗时: {end_time - start_time:.3f} 秒")
print("Perturb M_total =", M_total_p)

# ========================== 测试与误差评估 ==========================
Nx_test = 201
Ny_test = 201
x_test = np.linspace(-1, 1, Nx_test)
y_test = np.linspace(-1, 1, Ny_test)
xx, yy = np.meshgrid(x_test, y_test, indexing='xy')

# 预测主解
def predict_tanh(points, w, b, alpha):
    z = points @ w.T + b
    return np.tanh(z) @ alpha

def predict_sin(points, w, b, alpha):
    z = points @ w.T + b
    return np.sin(z) @ alpha

pts = np.column_stack([xx.ravel(), yy.ravel()])
# 根据象限选择对应网络
mask_mm = (pts[:,0] < 0) & (pts[:,1] < 0)
mask_pm = (pts[:,0] >= 0) & (pts[:,1] < 0)
mask_mp = (pts[:,0] < 0) & (pts[:,1] >= 0)
mask_pp = (pts[:,0] >= 0) & (pts[:,1] >= 0)

u_pred = np.zeros(len(pts))
u_pred[mask_mm] = predict_tanh(pts[mask_mm], w_mm, b_mm, alpha_mm)
u_pred[mask_pm] = predict_tanh(pts[mask_pm], w_pm, b_pm, alpha_pm)
u_pred[mask_mp] = predict_tanh(pts[mask_mp], w_mp, b_mp, alpha_mp)
u_pred[mask_pp] = predict_tanh(pts[mask_pp], w_pp, b_pp, alpha_pp)
u_pred = u_pred.reshape(xx.shape)

u_pred1 = np.zeros(len(pts))
u_pred1[mask_mm] = predict_sin(pts[mask_mm], w_mm1, b_mm1, alpha_mm_p)
u_pred1[mask_pm] = predict_sin(pts[mask_pm], w_pm1, b_pm1, alpha_pm_p)
u_pred1[mask_mp] = predict_sin(pts[mask_mp], w_mp1, b_mp1, alpha_mp_p)
u_pred1[mask_pp] = predict_sin(pts[mask_pp], w_pp1, b_pp1, alpha_pp_p)
u_pred1 = u_pred1.reshape(xx.shape)

# 精确解
u_exact = exact_u_piece(xx, yy)
error = u_exact - u_pred
error1 = error - epsilon * u_pred1

print(f"最大绝对误差: {np.max(np.abs(error)):.4e}")
print(f"相对L2误差: {np.linalg.norm(error)/np.linalg.norm(u_exact):.4e}")
print(f"Stage2相对L2误差: {np.linalg.norm(error1)/np.linalg.norm(u_exact):.4e}")
print(f"Stage2最大绝对误差: {np.max(np.abs(error1)):.4e}")