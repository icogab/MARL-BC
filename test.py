"""
Evaluates trained RL checkpoints on RBC simulators by running a deterministic
1 000-step rollout and saving the resulting time-series to a compressed `.npz`.

COMMAND-LINE USAGE
    python test.py <ALGO> <MODEL_TYPE|ALL> <N_AGENTS> <SAVE_MODE|ALL> [REP|ALL]

    ALGO         : PPO | SAC | DDPG | TD3 | A2C
    MODEL_TYPE   : KL | KL_HET | KL_TH | KS | KS_HET | KL_IRF | KL_TH_IRF | ALL
    N_AGENTS     : households/agents simulated
    SAVE_MODE    : best | linear | log | best_log | all
    REP          : run identifier (integer) or **all** to evaluate every replica

BEHAVIOUR
    ▸ Recursively scans *models/* to find checkpoints that match the filters.
    ▸ For each checkpoint:
          - loads the model,
          - builds (or reuses) a matching RBC environment,
          - performs a single deterministic episode of 1 000 steps,
          - records observations, actions, rewards and key simulator states.
    ▸ Writes results to `results/<ALGO>_<MODEL_TYPE>_<N_AGENTS>/rep<rep>/<SAVE_MODE>/`
      as `results_<checkpoint_name>[ _IRF].npz`, overwriting any previous file.

OUTPUTS
    ▸ `.npz` files containing NumPy arrays for time-series analysis
    ▸ Console logs with progress, episode length and save paths
"""


import os
import sys
import glob
import re
from typing import Dict, Tuple

import numpy as np
from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C

# ---------------------------------------------------------------------
#   Simulators + parallel env wrapper
# ---------------------------------------------------------------------
from simulators import (
    RBCSimulator_KL,
    RBCSimulator_KL_Heterogeneous,
    RBCSimulator_KS,
    RBCSimulator_KL_theory,
    RBCSimulator_KS_Heterogeneous,
    RBCSimulator_KL_IRF,
    RBCSimulator_KL_TH_IRF,
)
from gym_env import RBCParallelEnv

# ---------------------------------------------------------------------
#   Helpers: build / cache envs
# ---------------------------------------------------------------------

def _make_env(cfg: Dict) -> RBCParallelEnv:
    """Return a fresh RBCParallelEnv (no caching here)."""
    mtype = cfg["model_type"].upper()
    n_agents = cfg["n_agents"]

    # Observation subsets per model type
    if mtype.startswith("KL"):
        obs = ["capital"]
    elif mtype.startswith("KS_HET"):
        obs = [
            "capital", "mean_capital", "aggregate_state",
            "idiosyncratic_state", "capital_prod",
        ]
    elif mtype.startswith("KS"):
        obs = [
            "capital", "mean_capital", "aggregate_state",
            "idiosyncratic_state",
        ]
    elif mtype.startswith("K"):
        obs = ["capital", "TFP"]
    else:
        raise ValueError(f"Unrecognised model_type '{mtype}'")

    # Map model → simulator class
    sim_map = {
        "KL":            RBCSimulator_KL,
        "KL_HET":        RBCSimulator_KL_Heterogeneous,
        "KL_TH":         RBCSimulator_KL_theory,
        "KL_IRF":        RBCSimulator_KL_IRF,
        "KS":            RBCSimulator_KS,
        "KS_HET":        RBCSimulator_KS_Heterogeneous,
        "KL_TH_IRF":     RBCSimulator_KL_TH_IRF,
    }
    sim_class = sim_map.get(mtype)
    if sim_class is None:
        raise ValueError(f"model_type '{mtype}' not in simulator map")

    return RBCParallelEnv(
        simulator=sim_class(n_agents=n_agents),
        obs_space=obs,
        num_iters=cfg["num_iters"],
        render_mode=None,
    )

# Keep envs alive across replicas (SB3 models are deterministic given same env)
_env_cache: Dict[Tuple[str, int], RBCParallelEnv] = {}

def get_env(cfg: Dict) -> RBCParallelEnv:
    """Return cached env or create one."""
    key = (cfg["model_type"], cfg["n_agents"])
    if key not in _env_cache:
        _env_cache[key] = _make_env(cfg)
    return _env_cache[key]

# ---------------------------------------------------------------------
#   Roll-out helper
# ---------------------------------------------------------------------

def test_model_fast(model, cfg):
    env, tracked = _prepare_env_and_vars(cfg)
    obs, _ = env.reset()

    res = {v: [] for v in tracked + ["rewards", "obs", "actions"]}
    done = False

    while not done:
        batch = np.stack(list(obs.values()))          # SB3 wants ndarray
        a_batch, _ = model.predict(batch, deterministic=True)
        actions = {ag: a for ag, a in zip(obs.keys(), a_batch)}

        res["obs"].append({k: v.copy() for k, v in obs.items()})
        res["actions"].append(actions.copy())

        obs, rew, _, truncs, _ = env.step(actions)
        res["rewards"].append(rew.copy())

        sim = env.simulator
        for v in tracked:
            if v == "mean_capital":
                val = np.mean(sim.ks)
            elif v == "aggregate_state":
                val = sim.z_flag
            else:
                val = getattr(sim, v)
            res[v].append(val.copy() if isinstance(val, np.ndarray) else val)

        done = all(truncs.values())

    return res


def _prepare_env_and_vars(cfg):
    """Select which simulator variables we track."""
    env = get_env(cfg)

    base_vars = ["cs", "wealths", "incomes", "z", "Y", "A", "ks", "r", "K"]
    kl_vars   = ["w", "ls", "L"]
    kl_het    = ["KK", "LL", "cap_prods", "labor_prods"]
    ks_vars   = [
        "K", "u_z", "L", "Y", "r", "w", "incomes", "wealths",
        "cs", "ks", "eps", "z_flag", "mean_capital", "aggregate_state",
    ]

    mtype = cfg["model_type"].upper()
    if mtype.startswith("KL_HET"):
        tracked = base_vars + kl_vars + kl_het
    elif mtype.startswith("KL"):
        tracked = base_vars + kl_vars
    elif mtype.startswith("KS_HET"):
        tracked = ks_vars + ["cap_prods"]
    elif mtype.startswith("KS"):
        tracked = ks_vars
    else:
        tracked = base_vars
    return env, tracked

# ---------------------------------------------------------------------
#   MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        print("Usage: python test.py <ALGO> <MODEL_TYPE|ALL> <NUM_AGENTS> <SAVE_MODE|ALL> [REP_ID|ALL]")
        sys.exit(1)

    algo      = sys.argv[1].upper()
    model_sel = sys.argv[2].upper()
    n_agents  = int(sys.argv[3])
    save_arg  = sys.argv[4].lower()
    rep_arg   = sys.argv[5].lower() if len(sys.argv) == 6 else "all"

    algo_cls = {
        "PPO":  PPO,
        "SAC":  SAC,
        "DDPG": DDPG,
        "TD3":  TD3,
        "A2C":  A2C,
    }.get(algo)
    if algo_cls is None:
        print(f"Algorithm {algo} not supported.")
        sys.exit(1)

    valid_modes = {"linear", "log", "best", "best_log", "all"}
    if save_arg not in valid_modes:
        print(f"SAVE_MODE must be one of {valid_modes}.")
        sys.exit(1)

    # Modes to test
    save_modes = ["linear", "log", "best", "best_log"] if save_arg == "all" else [save_arg]

    # KL_IRF / KL_TH_IRF reuse the reference KL models when loading
    search_type = {
        "KL_IRF":    "KL",
        "KL_TH_IRF": "KL_TH",
    }.get(model_sel, model_sel)

    rep_pattern  = "*_rep*" if rep_arg == "all" else f"*_rep{rep_arg}"
    base_pattern = f"models/model_{algo}_num_agents_{n_agents}_{rep_pattern}"

    for model_dir in sorted(glob.glob(base_pattern)):
        if not os.path.isdir(model_dir):
            continue

        # Example: model_PPO_num_agents_50_KL_rep3 → dir_type == "KL"
        dir_type = os.path.basename(model_dir).split(
            f"num_agents_{n_agents}_", 1
        )[1].rsplit("_rep", 1)[0]

        if model_sel != "ALL" and dir_type != search_type:
            continue

        rep = os.path.basename(model_dir).rsplit("_rep", 1)[1]

        for mode in save_modes:
            res_base = os.path.join(
                "results",
                f"{algo}_{model_sel}_{n_agents}",
                f"rep{rep}",
                mode,
            )
            os.makedirs(res_base, exist_ok=True)

            model_paths = sorted(glob.glob(os.path.join(model_dir, mode, "*.zip")))
            if not model_paths:
                print(f"[skip] {mode} (rep{rep}): no .zip found")
                continue

            # best_log: de-duplicate same step
            if mode == "best_log":
                model_paths.sort(
                    key=lambda p: (0 if os.path.basename(p).startswith("best_") else 1, p)
                )

                tested_keys = set()
                def _key(p: str) -> str:
                    fname = os.path.splitext(os.path.basename(p))[0]
                    return re.sub(r"^(best_|log_)", "", fname)

                filtered = []
                for p in model_paths:
                    k = _key(p)
                    if k in tested_keys:
                        continue
                    tested_keys.add(k)
                    filtered.append(p)
                model_paths = filtered

            for model_path in model_paths:
                fname = os.path.splitext(os.path.basename(model_path))[0]
                suffix = "_IRF" if model_sel == "KL_IRF" else ""

                print(f"\n=== {algo}/{model_sel} rep{rep} [{mode}] — {fname}.zip ===")

                model = algo_cls.load(model_path)
                cfg = {
                    "n_agents":  n_agents,
                    "model_type": model_sel,
                    "num_iters": 1000,
                }
                out = test_model_fast(model, cfg)

                print(f"   Episode completed — total steps: {len(out['rewards'])}")

                # Overwrite (no _1, _2...)
                result_fname = f"results_{fname}{suffix}.npz"
                result_path  = os.path.join(res_base, result_fname)

                if os.path.exists(result_path):
                    print(f"   Overwriting existing file at {result_path}")

                np.savez_compressed(result_path, **out)
                print(f"   Saved → {result_path}")
