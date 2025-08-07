"""
Batch launcher for *training.py* and *test.py*

HOW TO RUN
    python launch_batch.py        # after editing the CONFIG block below

WHAT IT DOES
    • Pins every BLAS/NumPy backend to a single thread (stable timings).
    • Generates one job for every (ALGO, RBC_TYPE, N_AGENTS, SAVE_MODE, REP):
          - runs **training.py** first,
          - then **test.py** with the same arguments.
    • Writes stdout/err of each job to  jobs/<train|test>_<args>.txt.
    • Executes the whole queue with  parallelization.main(jobs, N_concurrent).

Edit the CONFIG section to change the experiment grid or parallelism level.
"""

import os
from parallelization import main

if __name__ == "__main__":

    # --- Force single-thread BLAS back-ends ---------------------------
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"

    # -----------------------------------------------------------------
    os.makedirs("jobs", exist_ok=True)
    jobs = []

    # ------------------------------- CONFIG -------------------------------

    algorithm_types = ["PPO", "SAC", "TD3"]      # RL algorithms to train/test
    rbc_types       = ["KS_HET"]                 # RBC simulator model type
    num_agents_list = [20]                       # Number of households per run
    save_modes      = ["best_log"]               # Checkpoint-saving strategy
    reps            = [0, 1, 2]                  # Replica IDs (repetition)
    N_concurrent    = 48                         # Max concurrent processes


    # --- Build TRAINING jobs -----------------------------------------
    for algo in algorithm_types:
        for rbc in rbc_types:
            for n_agents in num_agents_list:
                for mode in save_modes:
                    for rep in reps:
                        outfile = os.path.join(
                            "jobs",
                            f"train_{algo}_{rbc}_{n_agents}_{mode}_rep{rep}.txt",
                        )
                        cmd = [
                            "python", "training.py",
                            algo,
                            rbc,
                            str(n_agents),
                            mode,
                            str(rep),
                        ]
                        jobs.append({"cmd": cmd, "outfile": outfile})

    # --- Build TEST jobs ---------------------------------------------
    for algo in algorithm_types:
        for rbc in rbc_types:
            for n_agents in num_agents_list:
                for mode in save_modes:
                    for rep in reps:
                        outfile = os.path.join(
                            "jobs",
                            f"test_{algo}_{rbc}_{n_agents}_{mode}_rep{rep}.txt",
                        )
                        cmd = [
                            "python", "test.py",
                            algo,
                            rbc,
                            str(n_agents),
                            mode,
                            str(rep),
                        ]
                        jobs.append({"cmd": cmd, "outfile": outfile})

    # --- Run everything (set concurrency here) -----------------------
    N_concurrent = 48
    main(jobs, N_concurrent)
