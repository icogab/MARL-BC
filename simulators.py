"""
Minimal RBC simulators (KL & KS variants).

Change the baseline parameters once; all models inherit them.
"""

import numpy as np

# ------------------------------------------------------------------
# Baseline parameters ---------------------------
# ------------------------------------------------------------------
alpha          = 0.36   # Capital share in Cobb–Douglas production
delta          = 0.025  # Capital depreciation per period
rho            = 0.9    # Persistence of log‑TFP (AR(1))
sigma          = 0.01   # Std. dev. of TFP shock
gamma          = 1.0    # CRRA coefficient
u_g            = 0.04   # Unemployment in good state (KS)
u_b            = 0.10   # Unemployment in bad  state (KS)
l_bar          = 1.11   # Labour endowment per employed worker (KS)
zg             = 1.01   # TFP level, good state
zb             = 0.99   # TFP level, bad  state
low_cap_prod   = 0.25   # κ for low‑productivity group (KS‑het)
mid_cap_prod   = 1.0    # κ for middle group (KS‑het)
high_cap_prod  = 1.5    # κ for high‑productivity group (KS‑het)
Prod_GRID = np.array([0.98, 1.0, 1.02])  # κ, λ grid for KL‑het

# ------------------------------------------------------------------
# Utility functions ---------------------------------------------------
# ------------------------------------------------------------------

def _CRRA_utility_single(c, gamma=1.0):
    """Single‑argument CRRA utility with a heavy penalty for c ≤ 0."""
    if c <= 0:
        return -1e9
    if np.isclose(gamma, 1.0):
        return np.log(c)
    return (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def _CRRA_utility_cl(c, l, gamma=1.0, b=5.0):
    """Utility from consumption and leisure."""
    return _CRRA_utility_single(c, gamma) + b * _CRRA_utility_single(1.0 - l, gamma)

# ==================================================================
# KL MODELS  (capital + labour, mean‑field) -------------------------
# ==================================================================

class RBCSimulator_KL:
    """Homogeneous agents, endogenous labour supply (baseline KL)."""

    def __init__(self, alpha=alpha, delta=delta, rho=rho, sigma=sigma,
                 n_agents=1, gamma=1.0):
        # Parameters
        self.alpha = alpha
        self.delta = delta
        self.rho = rho
        self.sigma = sigma
        self.n = n_agents
        self.gamma = gamma
        self.reset()

    # --------------------------------------------------------------
    # Initialisation / reset ---------------------------------------
    # --------------------------------------------------------------
    def reset(self):
        # Individual state: capital k_i, labour l_i, consumption c_i
        self.ks = np.ones(self.n)
        self.ls = np.ones(self.n)
        self.cs = np.zeros(self.n)
        # Aggregate technology
        self.z= 0.0  # Log TFP
        self.A = 1.0  # TFP level
        self._update_prod()  # Compute initial prices & output

    # --------------------------------------------------------------
    # Technology and prices ----------------------------------------
    # --------------------------------------------------------------
    def _update_tfp(self):
        """AR(1) shock to log‑TFP."""
        self.z = self.rho * self.z + self.sigma * np.random.randn()
        self.A = np.exp(self.z)

    def _update_prod(self):
        """Aggregate production Y and factor demand."""
        self.K = np.mean(self.ks)
        self.L = np.mean(self.ls)
        self.Y = self.A * self.K ** self.alpha * self.L ** (1.0 - self.alpha)

    def factor_prices(self):
        """Compute factor prices r (capital) and w (wages)."""
        self.r = self.alpha * (self.Y / self.K)
        self.w = (1.0 - self.alpha) * (self.Y / self.L)

    def update_hh(self, actions):
        """Update household choices given actions = [c_frac_i … l_i]."""
        # Split action vector
        self.c_fracs = actions[:self.n]
        self.ls = actions[self.n:]

        # Perfect competition ⇒ zero profits
        self.incomes = self.r * self.ks + self.w * self.ls
        self.wealths = self.incomes + (1 - self.delta) * self.ks
        self.cs = self.wealths * self.c_fracs
        self.ks = self.wealths * (1 - self.c_fracs)

    # --------------------------------------------------------------
    # One‑period transition ----------------------------------------
    # --------------------------------------------------------------
    def step(self, actions):
        """Advance one period (actions length = 2n)."""
        self._update_tfp()      # 1. Draw TFP shock
        self._update_prod()     # 2. Update production
        self.factor_prices()    # 3. Compute prices
        self.update_hh(actions) # 4. Update households

    def get_utilities(self):
        """Return current‑period utilities for all agents."""
        return np.array([_CRRA_utility_cl(c, l, self.gamma) for c, l in zip(self.cs, self.ls)])

# ------------------------------------------------------------------
# KL Variants -------------------------------------------------------
# ------------------------------------------------------------------
class RBCSimulator_KL_theory(RBCSimulator_KL):
    """Full depreciation variant (δ = 1)."""
    def __init__(self, **kw):
        kw.setdefault('delta', 1.0)
        super().__init__(**kw)


class RBCSimulator_KL_Heterogeneous(RBCSimulator_KL):
    """Fixed heterogeneous productivities in capital (κ) and labour (λ)."""

    prod_grid = Prod_GRID  # Common grid for (κ, λ)

    def __init__(self, **kw):
        self.n = kw.get('n_agents', 2)
        # Assign (κ, λ) from 3×3 grid
        kap, lab = np.meshgrid(self.prod_grid, self.prod_grid)
        pairs = np.column_stack([kap.ravel(), lab.ravel()])
        if self.n <= 9:
            self.kappa, self.lambda_ = pairs[: self.n].T
        else:
            idx = np.random.choice(len(pairs), self.n, replace=True)
            self.kappa, self.lambda_ = pairs[idx].T
        super().__init__(**kw)

    def _update_prod(self):
        """Aggregate production with heterogeneous (κ, λ)."""
        self.KK = np.mean(self.ks * self.kappa)
        self.LL = np.mean(self.ls * self.lambda_)
        self.Y = self.A * self.KK ** self.alpha * self.LL ** (1.0 - self.alpha)

    def factor_prices(self):
        """Individual factor prices under heterogeneity."""
        self.r = (self.alpha / self.n) * (self.Y / self.KK) * self.kappa
        self.w = ((1 - self.alpha) / self.n) * (self.Y / self.LL) * self.lambda_

# ---------------------------------------------------------------------------------
# KL Variants for impulse‑response analysis ---------------------------------------
# ---------------------------------------------------------------------------------

class RBCSimulator_KL_IRF(RBCSimulator_KL):
    """Impulse‑response: one‑off TFP jump followed by AR(1) decay."""

    def __init__(self, t_shock=100, shock_size=0.1, **kw):
        self.t_shock, self.shock = t_shock, shock_size
        super().__init__(**kw)
        self.t = 0

    def _update_tfp(self):
        """Deterministic TFP path for IRF."""
        self.t += 1
        if self.t == self.t_shock:
            self.z = self.shock
        elif self.t > self.t_shock:
            self.z = self.rho * self.z
        else:
            self.z = 0.0
        self.A = np.exp(self.z)


class RBCSimulator_KL_TH_IRF(RBCSimulator_KL_IRF):
    """Full depreciation version."""
    def __init__(self, **kw):
        kw.setdefault('delta', 1.0)
        super().__init__(**kw)

# ==================================================================
# KS MODELS  (aggregate & idiosyncratic shocks) ---------------------
# ==================================================================

class RBCSimulator_KS_Heterogeneous:
    """KS model with heterogeneous κ: 15% low, 70% mid, 15% high."""

    def __init__(self, alpha, delta, n_agents=20, gamma=2,
                 zg=zg, zb=zb, u_g=u_g, u_b=u_b, l_bar=l_bar,
                 low_cap_prod=low_cap_prod, mid_cap_prod=mid_cap_prod, high_cap_prod=high_cap_prod,
                 P_z=None):

        # Core parameters
        self.alpha = alpha
        self.delta = delta
        self.n_agents = n_agents
        self.gamma = gamma

        # Aggregate technology
        self.z_vals = np.array([zg, zb])
        self.P_z = P_z if P_z is not None else np.array([[0.875, 0.125],
                                                         [0.125, 0.875]])

        # Labour market
        self.u_g, self.u_b = u_g, u_b
        self.l_bar = l_bar

        # Idiosyncratic employment transition matrices
        self.P_eps = {
            (0, 0): np.array([[0.97222222, 0.02777778],
                              [0.66666667, 0.33333333]]),
            (0, 1): np.array([[0.92708333, 0.07291667],
                              [0.25      , 0.75      ]]),
            (1, 0): np.array([[0.98333333, 0.01666667],
                              [0.75      , 0.25      ]]),
            (1, 1): np.array([[0.95555556, 0.04444444],
                              [0.4       , 0.6       ]])
        }

        # Heterogeneous capital productivity (κ)
        n_low  = max(1, int(np.ceil(0.15 * n_agents)))
        n_high = max(1, int(np.ceil(0.15 * n_agents)))
        n_mid  = n_agents - n_low - n_high
        self.kappa = np.concatenate([
            np.full(n_low,  low_cap_prod),
            np.full(n_mid,  mid_cap_prod),
            np.full(n_high, high_cap_prod)
        ])

        self.reset()  # Initialise state

    def reset(self):
        """Initialise or reset the economy."""
        self.ks = np.random.uniform(10, 70, size=self.n_agents)  # Capital stocks
        self.cs = np.zeros(self.n_agents)
        self.incomes = np.zeros(self.n_agents)
        self.wealths = self.ks.copy()

        self.eps = None  # Employment status
        self.z_flag = None  # Aggregate state index
        self.z = None
        self.K = None
        self.K_eff = None
        self.L = None
        self.Y = None
        self.r = None
        self.w = None
        self.u_z = None
        self._update_shocks()

    # ------------------------ internal helpers --------------------
    def _adjust_employment(self, eps, target_rate):
        """Force employment to match target unemployment rate."""
        target = int(round((1 - target_rate) * self.n_agents))
        gap = target - eps.sum()
        if gap > 0:
            zeros = np.where(eps == 0)[0]
            eps[np.random.choice(zeros, gap, replace=False)] = 1
        elif gap < 0:
            ones = np.where(eps == 1)[0]
            eps[np.random.choice(ones, -gap, replace=False)] = 0
        return eps

    def _update_shocks(self):
        """Draw aggregate and idiosyncratic shocks."""
        # Aggregate TFP state
        if self.z_flag is None:
            p01, p10 = self.P_z[0,1], self.P_z[1,0]
            pi = [p10/(p01+p10), p01/(p01+p10)]
            self.z_flag = np.random.choice([0,1], p=pi)
        else:
            old = self.z_flag
            self.z_flag = np.random.choice([0,1], p=self.P_z[old])
        self.z = self.z_vals[self.z_flag]

        # Employment status
        if self.eps is None:
            u0 = self.u_g if self.z_flag == 0 else self.u_b
            p_emp = 1 - u0
            eps = (np.random.rand(self.n_agents) < p_emp).astype(int)
            self.eps = self._adjust_employment(eps, u0)
        else:
            old = 1 - self.eps
            Pmat = self.P_eps[(1 - self.z_flag) ^ (1 - old), self.z_flag]  # Key = (old_z, new_z)
            row_idx = 1 - self.eps
            p_emp_vec = Pmat[row_idx, 0]
            new_eps = (np.random.rand(self.n_agents) < p_emp_vec).astype(int)
            u_now = self.u_g if self.z_flag == 0 else self.u_b
            self.eps = self._adjust_employment(new_eps, u_now)

    def update_production(self):
        """Compute aggregates and effective capital."""
        self.K = np.mean(self.ks)
        self.KK = np.mean(self.ks * self.kappa)  # Effective capital
        self.u_z = self.u_g if self.z_flag == 0 else self.u_b
        self.L = self.l_bar * (1 - self.u_z)
        self.Y = self.z * self.K_eff**self.alpha * self.L**(1 - self.alpha)

    def factor_prices(self):
        """Compute factor prices r and w."""
        self.r = self.alpha * self.Y / self.KK
        self.w = (1 - self.alpha) * self.Y / self.L

    def update_hh(self, actions):
        """Update individual states given consumption fractions."""
        self.c_fracs = actions
        self.incomes = self.r * self.ks + self.w * self.l_bar * self.eps
        self.wealths = self.incomes + (1 - self.delta) * self.ks
        self.cs = self.wealths * self.c_fracs
        self.ks = self.wealths * (1 - self.c_fracs)

    def step(self, actions):
        """Advance one period."""
        self._update_shocks()
        self.update_production()
        self.update_hh(actions)

    def get_utilities(self):
        """Return CRRA utilities."""
        return np.array([_CRRA_utility_single(c, self.gamma) for c in self.cs])


class RBCSimulator_KS(RBCSimulator_KS_Heterogeneous):
    """KS with homogeneous κ = 1."""
    def __init__(self, alpha, delta, n_agents=20, gamma=2,
                 zg=zg, zb=zb, u_g=u_g, u_b=u_b, l_bar=l_bar,
                 P_z=None):
        super().__init__(alpha=alpha, delta=delta, n_agents=n_agents,
                         gamma=gamma, zg=zg, zb=zb,
                         u_g=u_g, u_b=u_b, l_bar=l_bar,
                         low_cap_prod=1.0, mid_cap_prod=1.0, high_cap_prod=1.0,
                         P_z=P_z)
