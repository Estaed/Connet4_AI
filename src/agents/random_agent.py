"""
Random Agent for Connect4 RL System

This module implements a baseline random agent that selects valid moves
randomly. It serves as a benchmark for comparing more sophisticated agents
and provides a simple baseline for testing the environment.
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Random agent that selects valid moves uniformly at random.

    This agent serves multiple purposes:
    1. Baseline for performance comparison
    2. Environment testing and validation
    3. Opponent for training other agents
    4. Simple demonstration of the agent interface

    The agent makes no strategic decisions and has no learning capability,
    making it ideal for basic testing scenarios.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        name: Optional[str] = None,
        config_path: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize random agent.

        Args:
            device: Device to use (not relevant for random agent)
            name: Human-readable name for the agent
            config_path: Path to configuration file
            seed: Random seed for reproducible behavior
        """
        super().__init__(
            device=device, name=name or "RandomAgent", config_path=config_path
        )

        # Set random seed if provided
        self._seed: Optional[int] = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.logger.info(f"RandomAgent initialized with seed={seed}")

    def get_action(
        self,
        observation: np.ndarray,
        valid_actions: Optional[List[int]] = None,
        **kwargs,
    ) -> int:
        """
        Get random action from valid moves.

        Args:
            observation: Board state as 6x7 numpy array
                        (not used by random agent)
            valid_actions: List of valid column indices (0-6)
            **kwargs: Additional parameters (ignored)

        Returns:
            Random column index from valid actions

        Raises:
            ValueError: If no valid actions are provided or list is empty
        """
        # If no valid actions provided, assume all columns are valid
        if valid_actions is None:
            valid_actions = list(range(7))

        # Validate that we have valid actions
        if not valid_actions:
            raise ValueError(
                "No valid actions available for random agent"
            )

        # Ensure all actions are within valid range
        valid_actions = [
            action for action in valid_actions if 0 <= action <= 6
        ]

        if not valid_actions:
            raise ValueError("No valid actions in range 0-6")

        # Select random action
        action = random.choice(valid_actions)

        # Store for potential debugging/analysis
        self.last_observation = observation  # type: ignore
        self.last_action = action  # type: ignore

        self.logger.debug(
            f"RandomAgent selected action {action} from {valid_actions}"
        )

        return action

    def update(
        self, experiences: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, float]:
        """
        Update agent (no-op for random agent).

        Random agent has no learning capability, so this method
        does nothing and returns empty metrics.

        Args:
            experiences: Training experiences (ignored)
            **kwargs: Additional parameters (ignored)

        Returns:
            Empty dictionary (no training metrics for random agent)
        """
        # Random agent doesn't learn, so return empty metrics
        return {}

    def is_learning_agent(self) -> bool:
        """
        Check if this is a learning agent.

        Returns:
            False (random agent has no learning capability)
        """
        return False

    def reset_episode(self) -> None:
        """
        Reset agent state for new episode.

        For random agent, this just clears the last observation/action.
        """
        super().reset_episode()
        self.logger.debug("RandomAgent episode reset")

    def _get_save_state(self) -> Dict[str, Any]:
        """
        Get random agent specific state for saving.

        Returns:
            Dictionary containing random seed if set
        """
        state = super()._get_save_state()
        if self._seed is not None:
            state["seed"] = self._seed
        return state

    def _set_load_state(self, state: Dict[str, Any]) -> None:
        """
        Set random agent specific state from loaded data.

        Args:
            state: Loaded state dictionary
        """
        super()._set_load_state(state)

        # Restore random seed if it was saved
        if "seed" in state:
            self._seed = state["seed"]
            random.seed(self._seed)
            np.random.seed(self._seed)
            self.logger.info(
                f"RandomAgent restored with seed={self._seed}"
            )

    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information and metadata.

        Returns:
            Dictionary with agent information including seed
        """
        info = super().get_info()
        info["seed"] = self._seed
        info["strategy"] = "random_uniform"
        return info

    def __str__(self) -> str:
        """String representation of random agent."""
        stats = self.get_statistics()
        seed_info = (
            f" (seed={self._seed})" if self._seed is not None else ""
        )
        return (
            f"{self.name}{seed_info} "
            f"(Games: {stats['total_games']}, "
            f"Win Rate: {stats['win_rate']:.2%})"
        )


# Utility function for creating random agents with specific configurations
def create_seeded_random_agent(
    seed: int, name: Optional[str] = None
) -> RandomAgent:
    """
    Create a random agent with a specific seed for reproducible behavior.

    Args:
        seed: Random seed to use
        name: Optional name for the agent

    Returns:
        RandomAgent with specified seed
    """
    return RandomAgent(
        seed=seed, name=name or f"RandomAgent_seed_{seed}"
    )


def create_random_agent_pair(
    seed1: int, seed2: int
) -> tuple[RandomAgent, RandomAgent]:
    """
    Create two random agents with different seeds for testing.

    Args:
        seed1: Seed for first agent
        seed2: Seed for second agent

    Returns:
        Tuple of two RandomAgents with different seeds
    """
    agent1 = RandomAgent(
        seed=seed1, name=f"RandomAgent_1_seed_{seed1}"
    )
    agent2 = RandomAgent(
        seed=seed2, name=f"RandomAgent_2_seed_{seed2}"
    )
    return agent1, agent2
