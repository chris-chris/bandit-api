from typing import List

import numpy as np


class UCB1:
    def __init__(self, actions: List[str]):
        self.actions, n_actions = actions, len(actions)

        self.action_successes = np.zeros((n_actions))
        self.action_tries = np.zeros((n_actions))

    def _increment_action_tries(self, action: str) -> None:
        self.action_tries[self.actions.index(action)] += 1

    def _get_action_with_max_ucb(self) -> str:
        ucb_numerator = 2 * np.log(np.sum(self.action_tries))
        per_action_means = self.action_successes / self.action_tries
        ucb1_estimates = per_action_means + \
            np.sqrt(ucb_numerator / self.action_tries)

        return self.actions[np.nanargmax(ucb1_estimates)]

    def select_action(self) -> str:
        untested_actions = np.nonzero(self.action_tries == 0)[0]
        if untested_actions.size == 0:
            best_action_so_far = self._get_action_with_max_ucb()
            self._increment_action_tries(best_action_so_far)
            return best_action_so_far
        else:
            untested_action = self.actions[untested_actions[0]]
            self._increment_action_tries(untested_action)
            return untested_action

    def reward_action(self, action: str) -> None:
        if action not in self.actions:
            raise ValueError(f"action {action} not recognized")
        action_index = self.actions.index(action)
        self.action_successes[action_index] += 1
