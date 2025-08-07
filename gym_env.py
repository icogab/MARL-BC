"""
ParallelEnv wrapper for RBC simulators (KL / KS variants).

• Exposes each simulator as a PettingZoo parallel environment for
  Stable-Baselines3 + SuperSuit pipelines.
• User-selectable observation list (`obs_space`); 1-D action for KS,
  2-D (consumption, labour) for KL; actions auto-scaled to [0, 1].
• Handles any number N of heterogeneous agents and truncates each episode
  after `num_iters` steps.
• **Import** this class from training / testing scripts — it is not meant
  to be executed standalone.
"""

import functools
import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from simulators import (
    RBCSimulator_KS,
    RBCSimulator_KL,
    RBCSimulator_KL_theory,
    RBCSimulator_KL_Heterogeneous,
    RBCSimulator_KS_Heterogeneous,
    RBCSimulator_KL_IRF,
    RBCSimulator_KL_TH_IRF,
)


class RBCParallelEnv(ParallelEnv):

    metadata = {"render_modes": ["human"], "name": "heterogeneous_returns_rbc"}

    def __init__(
        self,
        simulator,
        obs_space=["wealth"],
        num_iters=500,
        render_mode=None,
    ):
        self.simulator = simulator
        self.num_iters = num_iters
        self.render_mode = render_mode

        # list of agent names:  h_0, h_1, …
        self.possible_agents = [f"h_{r}" for r in range(self.simulator.n_agents)]
        # convenience mapping: name → integer index
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        # Identify simulator flavour
        if isinstance(
            simulator,
            (
                RBCSimulator_KL,
                RBCSimulator_KL_theory,        # already covered by inheritance
                RBCSimulator_KL_Heterogeneous,
                RBCSimulator_KL_IRF,           # listed explicitly for clarity
                RBCSimulator_KL_TH_IRF,
            ),
        ):
            self.simulator_type = "RBCKL"

        elif isinstance(
            simulator,
            (
                RBCSimulator_KS,
                RBCSimulator_KS_Heterogeneous,
            ),
        ):
            self.simulator_type = "RBCKS"

        else:
            raise ValueError(f"Unknown simulator type: {simulator.__class__.__name__}")

        # Action-vector length
        if self.simulator_type in ("RBCKS"):
            self.action_len = 1           # consumption only
        elif self.simulator_type == "RBCKL":
            self.action_len = 2           # consumption + labour
        else:                             # safeguard
            raise ValueError("Internal error: simulator_type is invalid")

        self.obs_space = obs_space
        # self.obs_len = len(obs_space) + 1   # with linear “type”
        self.obs_len = len(obs_space)        # without linear “type”

    # ------------------------------------------------------------------ #
    # Spaces
    # ------------------------------------------------------------------ #
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Observation space for each agent: Box(0, 100, (obs_len,)).
        Originally: [capital, linear “type” ∈ 0-1].
        """
        return Box(low=0.0, high=100.0, shape=(self.obs_len,))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Action space for each agent.
        • shape (action_len,) with values in [-0.99, 0.99]  
          (later converted to fractions in [0, 1]).
        """
        return Box(low=-0.99, high=0.99, shape=(self.action_len,))

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self):
        """Basic text rendering."""
        import gymnasium

        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render() without specifying any render mode."
            )
            return

        if len(self.agents) == self.simulator.n_agents:
            string = ""
            for i, agent in enumerate(self.agents):
                string += f"Agent {i} (name={agent}): {self.state[agent]}\n"
        else:
            string = "Game over"
        print(string)

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        """
        Environment reset:
        • restore agent list
        • zero the step counter
        • reset the RBC simulator
        • build and return initial observations
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0

        # reset simulator with identical RBC parameters
        self.simulator.reset()

        observations = self._get_observations()
        self.state = observations
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    # ------------------------------------------------------------------ #
    # Observation helpers
    # ------------------------------------------------------------------ #
    def _get_observations_from_simulator(self, obs_string):
        """Pull the requested variable from the simulator."""
        if obs_string == "capital":
            return self.simulator.ks
        if obs_string == "wealth":
            return self.simulator.wealths          # broadcast scalar
        elif obs_string == "income":
            return self.simulator.incomes
        elif obs_string == "labour":
            return self.simulator.ls
        elif obs_string == "mean_capital":
            return np.full(self.simulator.n_agents, np.mean(self.simulator.ks))
        elif obs_string == "mean_K":
            return np.full(self.simulator.n_agents, self.simulator.K)
        elif obs_string == "aggregate_state":
            return np.full(self.simulator.n_agents, self.simulator.z_flag)
        elif obs_string == "TFP":
            return np.full(self.simulator.n_agents, self.simulator.A)
        elif obs_string == "idiosyncratic_state":
            return np.full(self.simulator.n_agents, self.simulator.eps)
        elif obs_string == "capital_prod":
            return self.simulator.cap_prods
        else:
            raise ValueError(f"Unknown observation string: {obs_string}")

    def _get_observations(self):
        """
        Build a dict of observations for every agent:
        currently: [capital, …] (linear “type” disabled/commented).
        """
        type_array = np.linspace(0.0, 1.0, self.simulator.n_agents)
        obs = {}
        for i, agent in enumerate(self.agents):
            obs[agent] = []
            # loop over obs_space elements and grab the corresponding variable
            for space in self.obs_space:
                space_obs = self._get_observations_from_simulator(space)[i]
                obs[agent].append(space_obs)

            # # add linear “type”
            # obs[agent].append(type_array[i])

        return obs

    # ------------------------------------------------------------------ #
    # Reward helper
    # ------------------------------------------------------------------ #
    def _get_rewards(self):
        """
        Compute CRRA utility-based reward (via simulator).
        Floors extremely low / NaN utilities to -1.0e5.
        """
        us = self.simulator.get_utilities()
        min_utility = -1.0e5
        us = np.where(us < min_utility, min_utility, us)
        us = np.where(np.isnan(us), min_utility, us)
        return {agent: us[i] for i, agent in enumerate(self.agents)}

    # ------------------------------------------------------------------ #
    # Action helper
    # ------------------------------------------------------------------ #
    def _get_action_array(self, actions):
        """Convert dict → NumPy array and rescale to [0, 1]."""
        actions_array = np.array([actions[agent] for agent in self.agents])
        if self.simulator_type in ("RBC", "RBCKS"):
            c_fracs = actions_array[:, 0]
            c_fracs = (c_fracs + 1.0) / 2.0
            actions_array = c_fracs
        elif self.simulator_type == "RBCKL":
            c_fracs = actions_array[:, 0]
            l_fracs = actions_array[:, 1]
            c_fracs = (c_fracs + 1.0) / 2.0
            l_fracs = (l_fracs + 1.0) / 2.0
            actions_array = np.concatenate((c_fracs, l_fracs), axis=0)
        else:
            raise ValueError("Unknown simulator type")
        return actions_array

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, actions):
        """
        One environment step:
        • convert dict-actions to array
        • normalise actions to [0, 1]
        • advance the RBC simulator
        • compute rewards
        • manage terminations / truncations
        • return obs, reward, etc.
        """
        if not actions:              # no agents left
            self.agents = []
            return {}, {}, {}, {}, {}

        # actions → array (shape (n_agents,))
        acts_array = self._get_action_array(actions)

        # advance simulator
        self.simulator.step(acts_array)

        # rewards
        rewards = self._get_rewards()

        # we do not use early terminal states
        terminations = {agent: False for agent in self.agents}

        # increment step
        self.num_moves += 1
        env_truncation = self.num_moves >= self.num_iters
        truncations = {agent: env_truncation for agent in self.agents}

        # updated observations
        observations = self._get_observations()
        self.state = observations

        infos = {agent: {} for agent in self.agents}

        # if limit reached, clear agent list
        if env_truncation:
            self.agents = []

        # render if requested
        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos


# ---------------------------------------------------------------------- #
# Debug block (commented out by default)
# ---------------------------------------------------------------------- #
# if __name__ == "__main__":
#     from simulators import RBCSimulator_KL
#     simulator = RBCSimulator_KL(n_agents=1)
#     env = RBCParallelEnv(simulator, num_iters=100)
#     env.reset()
#
#     action = {"h_0": np.array([0.5, 0.5])}
#     for i in range(5):
#         obs, rewards, term, trunc, info = env.step(action)
#         print(f"Step {i}: {obs}  {rewards}")
