import os
from typing import Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

__all__ = [
    "evaluate_policy_with_discount",
    "BoolDonesWrapper",
    "SaveOnBestTrainingRewardCallback",
    "LinearSaveCallback",
    "LogSaveCallback",
]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _format_reward(val: float) -> str:
    """Convert a float reward to a filename‑safe string.

    * keep the minus sign for negative values (e.g. "-0.23")
    * keep the decimal point as dot ("1.50" → "1.50")
      Modern filesystems handle dots fine and this keeps the metric readable.
    """
    return f"{val:.2f}"


def evaluate_policy_with_discount(
    model,
    env: VecEnv,
    n_eval_episodes: int = 10,
    gamma: float = 0.95,
    deterministic: bool = True,
) -> Tuple[float, float]:
    """Run *n_eval_episodes* in *env* and return the mean ± std of discounted returns."""

    discounted_returns: list[float] = []
    obs = env.reset()

    for _ in range(n_eval_episodes):
        done = np.zeros(env.num_envs, dtype=bool)
        returns = np.zeros(env.num_envs)
        step = 0

        while not np.all(done):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, _ = env.step(action)

            returns += (gamma ** step) * rewards
            step += 1
            done = np.logical_or(done, dones)

        discounted_returns.extend(returns.tolist())
        obs = env.reset()

    return float(np.mean(discounted_returns)), float(np.std(discounted_returns))


# -----------------------------------------------------------------------------
# VecEnv helper
# -----------------------------------------------------------------------------

class BoolDonesWrapper(VecEnvWrapper):
    """Cast *dones* to bool – fixes dtype issues coming from SuperSuit."""

    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        if dones.dtype != np.bool_:
            dones = dones.astype(np.bool_)
        return obs, rew, dones, infos


# -----------------------------------------------------------------------------
# Callbacks with unified naming convention
# -----------------------------------------------------------------------------

class _BaseSaveCallback(BaseCallback):
    """Shared logic for evaluation, logging and saving."""

    prefix: str = "model"  # subclasses override

    def __init__(
        self,
        eval_env: VecEnv | gym.Env,
        save_path: str,
        n_eval_episodes: int,
        gamma: float,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.n_eval_episodes = n_eval_episodes
        self.gamma = gamma

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _eval_once(self) -> Tuple[float, float]:
        return evaluate_policy_with_discount(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            gamma=self.gamma,
            deterministic=True,
        )

    def _build_filename(self, step: int, mean_reward: float) -> str:
        rwd_str = _format_reward(mean_reward)
        return f"{self.prefix}_val_{rwd_str}_step_{step}.zip"

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)


class SaveOnBestTrainingRewardCallback(_BaseSaveCallback):
    """Save whenever we beat the best mean reward (with CI buffer)."""

    prefix = "best_model"

    def __init__(
        self,
        eval_env: VecEnv | gym.Env,
        save_path: str,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        gamma: float = 0.95,
        verbose: int = 0,
    ):
        super().__init__(eval_env, save_path, n_eval_episodes, gamma, verbose)
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_r, std_r = self._eval_once()
            if self.verbose:
                print(
                    f"[Best Eval] step={self.num_timesteps} mean={mean_r:.2f} ±{std_r:.2f} (best={self.best_mean_reward:.2f})"
                )
            st_err = std_r / np.sqrt(self.n_eval_episodes)
            if mean_r + 1.5 * st_err > self.best_mean_reward:
                if mean_r > self.best_mean_reward:
                    self.best_mean_reward = mean_r
                fname = self._build_filename(self.num_timesteps, mean_r)
                self.model.save(os.path.join(self.save_path, fname))
                if self.verbose:
                    print(f"[Best Save] {fname}")
            self.logger.record("eval/mean_reward", mean_r)
            self.logger.dump(self.num_timesteps)
        return True


class LinearSaveCallback(_BaseSaveCallback):
    """Evaluate and save every *save_freq* steps (until *max_steps*)."""

    prefix = "linear_model"

    def __init__(
        self,
        eval_env: VecEnv,
        save_path: str,
        save_freq: int,
        n_eval_episodes: int = 5,
        gamma: float = 0.99,
        max_steps: int = 2_000_000,
        verbose: int = 0,
    ):
        super().__init__(eval_env, save_path, n_eval_episodes, gamma, verbose)
        self.save_freq = save_freq
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        if self.num_timesteps <= self.max_steps and self.num_timesteps % self.save_freq == 0:
            mean_r, std_r = self._eval_once()
            if self.verbose:
                print(
                    f"[Linear Eval] step={self.num_timesteps} mean={mean_r:.2f} ±{std_r:.2f}"
                )
            fname = self._build_filename(self.num_timesteps, mean_r)
            self.model.save(os.path.join(self.save_path, fname))
            self.logger.record("eval/mean_reward", mean_r)
            self.logger.dump(self.num_timesteps)
        return True


class LogSaveCallback(_BaseSaveCallback):
    """Save at exponentially growing steps: initial × base^k."""

    prefix = "log_model"

    def __init__(
        self,
        eval_env: VecEnv,
        save_path: str,
        initial: int = 2,
        base: int = 2,
        n_eval_episodes: int = 5,
        gamma: float = 0.99,
        max_steps: int = 2_048_000,
        verbose: int = 0,
    ):
        super().__init__(eval_env, save_path, n_eval_episodes, gamma, verbose)
        self.next_threshold = initial
        self.base = base
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_threshold and self.next_threshold <= self.max_steps:
            mean_r, std_r = self._eval_once()
            if self.verbose:
                print(
                    f"[Log Eval] step={self.next_threshold} mean={mean_r:.2f} ±{std_r:.2f}"
                )
            fname = self._build_filename(self.next_threshold, mean_r)
            self.model.save(os.path.join(self.save_path, fname))
            self.logger.record("eval/mean_reward", mean_r)
            self.logger.dump(self.num_timesteps)
            self.next_threshold *= self.base
        return True
