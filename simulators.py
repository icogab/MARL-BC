"""Minimal RBC simulators for MARL-BC – KL & KS families.

Baseline parameters live at the top; tweak them once, and every model
below inherits the new values automatically.
"""

import numpy as np

# ------------------------------------------------------------------
# Baseline parameters (lower-case globals) --------------------------
# ------------------------------------------------------------------
alpha          = 0.36   # Cobb–Douglas capital share
delta          = 0.025  # depreciation rate per period
rho            = 0.9    # AR(1) persistence of log-TFP
sigma          = 0.01   # std. dev. of TFP shock
gamma          = 1.0    # CRRA coefficient (risk aversion)
u_g            = 0.04   # unemployment in good state (KS)
u_b            = 0.10   # unemployment in bad  state (KS)
l_bar          = 1.11   # labour endowment per employed worker (KS)
zg             = 1.01   # TFP level in good state
zb             = 0.99   # TFP level in bad  state
low_cap_prod   = 0.25   # κ for low-productivity group (KS-het)
mid_cap_prod   = 1.0    # κ for middle group (KS-het)
high_cap_prod  = 1.5    # κ for high-productivity group (KS-het)
Prod_GRID = np.array([0.98, 1.0, 1.02]) # κ  and λ grid for KL-HET

# ------------------------------------------------------------------
# Utility helpers ---------------------------------------------------
# ------------------------------------------------------------------

def _CRRA_utility_single(c, gamma=1.0):
    # CRRA utility with corner protection for c ≤ 0
    if c <= 0:
        return -1e9
    if np.isclose(gamma, 1.0):
        return np.log(c)
    return (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

def _CRRA_utility_cl(c, l, gamma=1.0, b=5.0):
    # Additive utility from consumption and leisure
    return _CRRA_utility_single(c, gamma) + b * _CRRA_utility_single(1.0 - l, gamma)

# ==================================================================
# KL MODELS  (capital + labour, mean-field)
# ==================================================================

class RBCSimulator_KL:
    """Baseline KL: homogeneous agents with labour supply."""

    def __init__(self, alpha=alpha, delta=delta, rho=rho, sigma=sigma,
                 n_agents=1, gamma=1.0):
        # parameters
        self.alpha, self.delta = alpha, delta
        self.rho, self.sigma = rho, sigma
        self.n, self.gamma = n_agents, gamma
        self.reset()

    # --------------------------------------------------------------
    # Initialisation / reset
    # --------------------------------------------------------------
    def reset(self):
        # capital k_i, labour l_i, consumption c_i (vectors length n)
        self.ks = np.ones(self.n)
        self.ls = np.ones(self.n)
        self.cs = np.zeros(self.n)
        # technology
        self.z, self.A = 0.0, 1.0
        self._update_prod()  # compute initial prices & output

    # --------------------------------------------------------------
    # Technology and prices
    # --------------------------------------------------------------
    def _update_tfp(self):
        # AR(1) shock to log-TFP
        self.z = self.rho * self.z + self.sigma * np.random.randn()
        self.A = np.exp(self.z)

    def _update_prod(self):
        # Aggregate production and factor prices
        self.K = np.mean(self.ks)
        self.L = np.mean(self.ls)
        self.Y = self.A * self.K ** self.alpha * self.L ** (1.0 - self.alpha)

    def factor_prices(self):
        """Compute factor prices r and w."""
        self.r = self.alpha * (self.Y / self.K)
        self.w = (1.0 - self.alpha) * (self.Y / self.L)


    def update_hh(self, actions):
        # Unpack delle azioni
        self.c_fracs = actions[:self.n]
        self.ls = actions[self.n:]
        
         # no profit in this economy with perfect competition
        self.incomes = self.r * self.ks + self.w * self.ls

        self.wealths = self.incomes + (1-self.delta) * self.ks

        self.cs = self.wealths * self.c_fracs
        self.ks = self.wealths * (1 - self.c_fracs)

    # --------------------------------------------------------------
    # One-period transition
    # --------------------------------------------------------------
    def step(self, actions):
        """Advance one period – actions = [c_frac_i … l_i] (len = 2n)."""

        # 1. shock and TFP update
        self._update_tfp()

        # 2. production and factor prices
        self._update_prod()

        # 3. update household choices
        self.factor_prices()

        # 4. update household incomes and choices
        self.update_hh(actions)


    def get_utilities(self):
        return np.array([_CRRA_utility_cl(c, l, self.gamma) for c, l in zip(self.cs, self.ls)])

# ------------------------------------------------------------------
# KL Variants
# ------------------------------------------------------------------
class RBCSimulator_KL_theory(RBCSimulator_KL):
    """Full depreciation (δ = 1)."""
    def __init__(self, **kw):
        kw.setdefault('delta', 1.0)
        super().__init__(**kw)

class RBCSimulator_KL_Heterogeneous(RBCSimulator_KL):
    """Fixed heterogeneity in capital & labour productivities."""
    prod_grid = Prod_GRID  # κ and λ grid for KL-HET

    def __init__(self, **kw):
        self.n = kw.get('n_agents', 2)
        # assign (κ_i, λ_i) from 3×3 grid
        kap, lab = np.meshgrid(self.prod_grid, self.prod_grid)
        pairs = np.column_stack([kap.ravel(), lab.ravel()])
        if self.n <= 9:
            self.kappa, self.lambda_ = pairs[: self.n].T
        else:
            idx = np.random.choice(len(pairs), self.n, replace=True)
            self.kappa, self.lambda_ = pairs[idx].T
        super().__init__(**kw)

    def _update_prod(self):
        self.KK = np.mean(self.ks * self.kappa)
        self.LL = np.mean(self.ls * self.lambda_)
        self.Y = self.A * self.KK ** self.alpha * self.LL ** (1.0 - self.alpha)


    def factor_prices(self):
        # compute individual factor prices
        self.r = (self.alpha / self.n) * (self.Y / self.KK) * self.kappa
        self.w = ((1 - self.alpha) / self.n) * (self.Y / self.LL) * self.lambda_


# ---------------------------------------------------------------------------------
# KL Variants for IRF
# ---------------------------------------------------------------------------------

class RBCSimulator_KL_IRF(RBCSimulator_KL):
    """Impulse-response: one-time TFP jump then AR(1) decay."""
    def __init__(self, t_shock=100, shock_size=0.1, **kw):
        self.t_shock, self.shock = t_shock, shock_size
        super().__init__(**kw)
        self.t = 0

    def _update_tfp(self):
        self.t += 1
        if self.t == self.t_shock:
            self.z = self.shock
        elif self.t > self.t_shock:
            self.z = self.rho * self.z
        else:
            self.z = 0.0
        self.A = np.exp(self.z)

class RBCSimulator_KL_TH_IRF(RBCSimulator_KL_IRF):
    def __init__(self, **kw):
        kw.setdefault('delta', 1.0)
        super().__init__(**kw)





# ==================================================================
# KS MODELS  (aggregate & idiosyncratic shocks)
# ==================================================================


class RBCSimulator_KS_Heterogeneous:
    """
    KS with heterogeneous κᵢ assigned by population quantiles: 15% low, 70% mid, 15% high.
    """
    def __init__(self, alpha, delta, n_agents=20, gamma=2,
                 zg=zg, zb=zb, u_g=u_g, u_b=u_b, l_bar=l_bar,
                 low_cap_prod=low_cap_prod, mid_cap_prod=mid_cap_prod, high_cap_prod=high_cap_prod,
                 P_z=None):
        
        # --- core parameters ------------------------------------------
        self.alpha = alpha
        self.delta = delta
        self.n_agents = n_agents
        self.gamma = gamma

        # --- aggregate technology -------------------------------------
        self.z_vals = np.array([zg, zb])
        self.P_z = P_z if P_z is not None else np.array([[0.875, 0.125],
                                                         [0.125, 0.875]])

        # --- labor market ---------------------------------------------
        self.u_g, self.u_b = u_g, u_b
        self.l_bar = l_bar

        # --- idiosyncratic transition matrices ------------------------
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

        # --- heterogeneous capital productivity ----------------------
        n_low  = max(1, int(np.ceil(0.15 * n_agents)))
        n_high = max(1, int(np.ceil(0.15 * n_agents)))
        n_mid  = n_agents - n_low - n_high
        self.kappa = np.concatenate([
            np.full(n_low,  low_cap_prod),
            np.full(n_mid,  mid_cap_prod),
            np.full(n_high, high_cap_prod)
        ])

        # --- initialize state ----------------------------------------
        self.reset()


    def reset(self):
        """Initialize or reset the economy state."""
        # capital k_i, labour l_i, consumption c_i (vectors length n)
        self.ks = np.random.uniform(10, 70, size=self.n_agents)
        self.cs = np.zeros(self.n_agents)
        self.incomes = np.zeros(self.n_agents)
        self.wealths = self.ks.copy()

        self.eps = None
        self.z_flag = None
        self.z = None
        self.K = None
        self.K_eff = None
        self.L = None
        self.Y = None
        self.r = None
        self.w = None
        self.u_z = None
        self._update_shocks()


    def _adjust_employment(self, eps, target_rate):
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
        """Updates z_flag, z and individual employment eps."""
        # aggregate shock
        if self.z_flag is None:
            p01, p10 = self.P_z[0,1], self.P_z[1,0]
            pi = [p10/(p01+p10), p01/(p01+p10)]
            self.z_flag = np.random.choice([0,1], p=pi)
        else:
            old = self.z_flag
            self.z_flag = np.random.choice([0,1], p=self.P_z[old])
        self.z = self.z_vals[self.z_flag]

        # idiosyncratic employment
        if self.eps is None:
            u0 = self.u_g if self.z_flag == 0 else self.u_b
            p_emp = 1 - u0
            eps = (np.random.rand(self.n_agents) < p_emp).astype(int)
            self.eps = self._adjust_employment(eps, u0)
        else:
            old = 1 - self.eps
            Pmat = self.P_eps[(1 - self.z_flag) ^ (1 - old), self.z_flag]  # using keys (old_z,new_z)
            row_idx = 1 - self.eps
            p_emp_vec = Pmat[row_idx, 0]
            new_eps = (np.random.rand(self.n_agents) < p_emp_vec).astype(int)
            u_now = self.u_g if self.z_flag == 0 else self.u_b
            self.eps = self._adjust_employment(new_eps, u_now)


    def update_production(self):
        """Compute aggregates and effective capital."""

        self.K = np.mean(self.ks)
        # effective capital: weighted average of individual capital
        self.KK = np.mean(self.ks * self.kappa)

        # unemployment rate based on current z_flag
        self.u_z = self.u_g if self.z_flag == 0 else self.u_b
        self.L = self.l_bar * (1 - self.u_z)

        # aggregate production
        self.Y = self.z * self.K_eff**self.alpha * self.L**(1 - self.alpha)


    def factor_prices(self):
        """Compute factor prices r and w based on current production."""
        self.r = self.alpha * self.Y / self.KK
        self.w = (1 - self.alpha) * self.Y / self.L


    def update_hh(self, actions):
        """Update individual incomes, wealths and consumption."""

        self.c_fracs = actions

        self.incomes = self.r * self.ks + self.w * self.l_bar * self.eps

        self.wealths = self.incomes + (1 - self.delta) * self.ks

        self.cs = self.wealths * self.c_fracs
        self.ks = self.wealths * (1 - self.c_fracs)


    def step(self, actions):
        """Advance one period given consumption fractions."""

        self._update_shocks()

        self.update_production()

        self.update_hh(actions)


    def get_utilities(self):
        """Return current-period CRRA utilities."""
        return np.array([_CRRA_utility_single(c, self.gamma) for c in self.cs])


class RBCSimulator_KS(RBCSimulator_KS_Heterogeneous):
    """
    KS as a special case of heterogeneous with κᵢ = 1 for all agents.
    """
    def __init__(self, alpha, delta, n_agents=20, gamma=2,
                 zg=zg, zb=zb, u_g=u_g, u_b=u_b, l_bar=l_bar,
                 P_z=None):
        
        # Use the same parameters as in the heterogeneous case,
        # but set κᵢ = 1 for all agents.
        super().__init__(alpha=alpha, delta=delta, n_agents=n_agents,
                         gamma=gamma, zg=zg, zb=zb,
                         u_g=u_g, u_b=u_b, l_bar=l_bar,
                         low_cap_prod=1.0, mid_cap_prod=1.0, high_cap_prod=1.0,
                         P_z=P_z)
