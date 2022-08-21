from typing import List

import numpy as np


class LinUCB:
    def __init__(self, actions: List[str], n_features: int, alpha: float):
        self.actions, self.n_actions = actions, len(actions)
        self.alpha = alpha

        self.action_tries = np.zeros((self.n_actions))

        self.covariance_matrices = np.tile(
            np.identity(n_features), (self.n_actions, 1, 1)
        )
        self.reward_matrix = np.zeros((self.n_actions, n_features))

    def _increment_action_tries(self, action: str) -> None:
        self.action_tries[self.actions.index(action)] += 1

    def _update_covariance_matrix(self, action: str, context: np.ndarray) -> None:
        action_index = self.actions.index(action)
        self.covariance_matrices[action_index] = self.covariance_matrices[
            action_index
        ] + np.dot(context, context.T)

    def _get_action_with_max_ucb(self, context: np.ndarray) -> str:
        inverse_covariance_matrices = np.linalg.inv(self.covariance_matrices)
        linucb_estimates = []
        for covariance_matrix, reward_vector in zip(
            inverse_covariance_matrices, self.reward_matrix
        ):
            arm_coefficients = np.dot(covariance_matrix, reward_vector)
            pointwise_estimate = np.dot(arm_coefficients, context)

            upper_bound = self.alpha * np.sqrt(
                np.dot(np.dot(context.T, covariance_matrix), context)
            )

            linucb_estimates.append(pointwise_estimate + upper_bound)

        return self.actions[np.nanargmax(linucb_estimates)]

    def select_action(self, context: np.ndarray) -> str:
        untested_actions = np.nonzero(self.action_tries == 0)[0]
        if untested_actions.size == 0:
            best_action_so_far = self._get_action_with_max_ucb(context)
            self._increment_action_tries(best_action_so_far)
            self._update_covariance_matrix(best_action_so_far, context)
            return best_action_so_far
        else:
            untested_action = self.actions[untested_actions[0]]
            self._increment_action_tries(untested_action)
            self._update_covariance_matrix(untested_action, context)
            return untested_action

    def reward_action(
        self, action: str, context: np.ndarray, reward: float = 1.0
    ) -> None:
        if action not in self.actions:
            raise ValueError(f"action {action} not recognized")
        action_index = self.actions.index(action)
        self.reward_matrix[action_index] += reward * context


if __name__ == "__main__":
    linucb = LinUCB(["a", "b", "c"], 3, 0.1)
    context = np.array([1, 0, 0])
    print(linucb.covariance_matrices)
    print(linucb.reward_matrix)

    action = linucb.select_action(context)
    print(f"{action} for context({context})")

    context = np.array([1, 1, 0])
    linucb.reward_action(action, context, 1)
    print(linucb.covariance_matrices)
    print(linucb.reward_matrix)

    context = np.array([0, 1, 1])
    action = linucb.select_action(context)
    print(f"{action} for context({context})")
    print(linucb.covariance_matrices)
    print(linucb.reward_matrix)

    context = np.array([0, 0, 1])
    action = linucb.select_action(context)
    print(f"{action} for context({context})")
    linucb.reward_action(action, context, 1)
    print(linucb.covariance_matrices)
    print(linucb.reward_matrix)

    context = np.array([1, 1, 1])
    action = linucb.select_action(context)
    print(f"{action} for context({context})")
    linucb.reward_action(action, context, 1)
    print(linucb.covariance_matrices)
    print(linucb.reward_matrix)
