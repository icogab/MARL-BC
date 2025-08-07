"""
Trains RL agents (PPO, SAC, DDPG, TD3, A2C) on RBC simulators (KS/KL).

Command-line usage
    python training.py <ALGO> <RBC_TYPE> <N_AGENTS> <SAVE_MODE> <REP>

    ALGO       : PPO | SAC | DDPG | TD3 | A2C
    RBC_TYPE   : KL | KL_HET | KL_TH | KS_HET | KS
    N_AGENTS   : number of households/agents in the simulator
    SAVE_MODE  : best | linear | log | best_log   (checkpointing strategy)
    REP        : integer tag to distinguish repeated runs

OUTPUTS
    ▸ Checkpoints & models:  models/<description>/
    ▸ TensorBoard logs:      tb_log/<description>/
"""



import sys, os
from typing import Type
from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
import supersuit as ss

from gym_env import RBCParallelEnv

from simulators import (
    RBCSimulator_KS,
    RBCSimulator_KL,
    RBCSimulator_KL_theory,
    RBCSimulator_KL_Heterogeneous,
    RBCSimulator_KS_Heterogeneous,
)

from utils import (
    SaveOnBestTrainingRewardCallback,
    LinearSaveCallback,
    LogSaveCallback,
)

# ------------------------------------------------------------------
# utilities
# ------------------------------------------------------------------

# Force every BLAS backend to a single core; reproducible wall‑times, no oversubscription.

def set_num_threads():
    os.environ.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

# Map the CLI identifier to the matching Stable‑Baselines3 class.

def get_algo_class(name: str) -> Type:
    return {"PPO": PPO, "SAC": SAC, "DDPG": DDPG, "TD3": TD3, "A2C": A2C}[name]

# Translate the requested RBC flavour into its simulator implementation.

def get_sim_class(t: str) -> Type:
    return {
        "KS": RBCSimulator_KS,
        "KL": RBCSimulator_KL,
        "KL_HET": RBCSimulator_KL_Heterogeneous,
        "KL_TH": RBCSimulator_KL_theory,
        "KS_HET": RBCSimulator_KS_Heterogeneous,
    }[t]

# Build a PettingZoo env, duplicate it n times, wrap with VecMonitor.

def make_vec_env(sim, obs, n):
    env = RBCParallelEnv(simulator=sim, obs_space=obs, num_iters=500)
    v = ss.pettingzoo_env_to_vec_env_v1(env)
    v = ss.concat_vec_envs_v1(v, num_vec_envs=n, num_cpus=n, base_class="stable_baselines3")
    return VecMonitor(v)

# Determine which state variables each agent observes for the chosen model.

def get_obs_space(rbc_type: str) -> list[str]:

    if rbc_type in {"K", "K_TH", "K_HET", "K_DIS"}:
        return ["capital"]
    
    if rbc_type == "K_IN":
        return ["income"]
    
    if rbc_type in {"KL", "KL_TH", "KL_HET"}:
        return ["capital"]
    
    if rbc_type in {"KS", "KS_LM"}:
        return ["capital", 
                "mean_capital", 
                "aggregate_state",
                "idiosyncratic_state"]
    
    if rbc_type == "KS_HET":
        return [
            "capital",
            "mean_capital",
            "aggregate_state",
            "idiosyncratic_state",
            "capital_prod",
        ]
    
    raise ValueError(f"Unknown RBC type {rbc_type}. Use 'K', 'KL', 'KS' or 'KS_LM'.")

# ------------------------------------------------------------------
# training loop
# ------------------------------------------------------------------

# Main entry called from CLI; sets up env, agent, callbacks, logging.

def train_rbc(
    model_name: str,
    rbc_type: str,
    num_agents: int,
    save_mode: str,
    n_envs: int = 2,
    rep: int = 2,
    total_timesteps: int = 2 ** 24,
):
    set_num_threads()

    ObsSpace = get_obs_space(rbc_type)
    Algo = get_algo_class(model_name)
    Sim = get_sim_class(rbc_type)
    simulator = Sim(n_agents=num_agents)

    vec_env = make_vec_env(simulator, ObsSpace, n_envs)
    vec_eval_env = make_vec_env(simulator, ObsSpace, 1)

    # Compose unique tags for logs and checkpoints.
    base_core = f"model_{model_name}_num_agents_{num_agents}_{rbc_type}_rep{rep}"
    iter_tag = f"_iters{total_timesteps}"
    base_name = base_core + iter_tag
    root_dir = "models"
    save_dir = os.path.join(root_dir, base_core, save_mode)
    os.makedirs(save_dir, exist_ok=True)

    # Discount factor: heterogeneous agents (KS) need 0.99, others run with 0.95.
    gamma = 0.99 if rbc_type.startswith("KS") else 0.95

    # Choose checkpointing strategy based on --save_mode flag.
    if save_mode == "best":
        callback = SaveOnBestTrainingRewardCallback(
            eval_env=vec_eval_env,
            save_path=save_dir,
            eval_freq=10_000,
            n_eval_episodes=100,
            verbose=1,
            gamma=gamma,
        )
    elif save_mode == "linear":
        callback = LinearSaveCallback(
            eval_env=vec_eval_env,
            save_path=save_dir,
            save_freq=50_000,
            n_eval_episodes=100,
            gamma=gamma,
            max_steps=total_timesteps,
            verbose=1,
        )
    elif save_mode == "log":
        callback = LogSaveCallback(
            eval_env=vec_eval_env,
            save_path=save_dir,
            initial=2,
            base=2,
            n_eval_episodes=100,
            gamma=gamma,
            max_steps=total_timesteps,
            verbose=1,
        )
    elif save_mode == "best_log":
        callback_1 = SaveOnBestTrainingRewardCallback(
            eval_env=vec_eval_env,
            save_path=save_dir,
            eval_freq=10_000,
            n_eval_episodes=100,
            verbose=1,
            gamma=gamma,
        )
        callback_2 = LogSaveCallback(
            eval_env=vec_eval_env,
            save_path=save_dir,
            initial=2,
            base=2,
            n_eval_episodes=30,
            gamma=gamma,
            max_steps=total_timesteps,
            verbose=1,
        )
        callback = [callback_1, callback_2]
    else:
        raise ValueError(f"Unknown save mode: {save_mode}")

    # Instantiate learner with simple MLP policy; state is low‑dimensional.
    model = Algo(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join("tb_log", base_core),
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(os.path.join(root_dir, base_core, base_name))
    print(f"Final checkpoint: models/{base_name}.zip")

    vec_env.close()
    vec_eval_env.close()
    print("Training finished — environments closed.")

# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(
            "Usage: python training.py <MODEL_NAME> <RBC_TYPE> <NUM_AGENTS> <SAVE_MODE> <REP>"
        )
        sys.exit(1)

    algo = sys.argv[1].upper()
    rtype = sys.argv[2].upper()
    nag = int(sys.argv[3])
    save_mode = sys.argv[4].lower()  # 'linear', 'log', 'best' or 'best_log'
    rep = int(sys.argv[5])

    try:
        train_rbc(algo, rtype, nag, save_mode=save_mode, rep=rep)
    except Exception as e:
        print(f"Errore: {e}")
        sys.exit(1)
