from typing import List

import random

import numpy as np


class EGreedy:
    def __init__(self, actions: List[str], epsilon: float = 0.1):
        self.actions, n_actions = actions, len(actions)
        self.epsilon = epsilon

        self.action_successes = np.zeros((n_actions))
        self.action_tries = np.zeros((n_actions))

    def _increment_action_tries(self, action: str) -> None:
        self.action_tries[self.actions.index(action)] += 1

    def _epsilon_greedy_selection(self):
        if random.random() < self.epsilon:
            random_action = random.choice(self.actions)
            return random_action
        else:
            best_action_so_far = self.actions[
                np.nanargmax(self.action_successes / self.action_tries)
            ]
            return best_action_so_far

    def select_action(self) -> str:
        untested_actions = np.nonzero(self.action_tries == 0)[0]
        if untested_actions.size == 0:
            epsilon_greedy_selection = self._epsilon_greedy_selection()
            self._increment_action_tries(epsilon_greedy_selection)
            return epsilon_greedy_selection
        else:
            untested_action = self.actions[untested_actions[0]]
            self._increment_action_tries(untested_action)
            return untested_action

    def reward_action(self, action: str) -> None:
        if action not in self.actions:
            raise ValueError(f"action {action} not recognized")
        action_index = self.actions.index(action)
        self.action_successes[action_index] += 1


if __name__ == "__main__":
    egreedy = EGreedy(["a", "b", "c"], 0.2)

    # 1
    action = egreedy.select_action()
    print(action)
    egreedy.reward_action(action)

    # 2
    action = egreedy.select_action()
    print(action)
    egreedy.reward_action(action)

    # 3
    action = egreedy.select_action()
    print(action)

    # 4
    action = egreedy.select_action()
    print(action)

    for _ in range(100):
        action = egreedy.select_action()
        print(action)
        if random.random() > 0.8:
            egreedy.reward_action(action)
            print("rewarded")

    print(egreedy.action_successes)
    print(egreedy.action_tries)
