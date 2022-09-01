from typing import List

import ujson
from aioredis import Redis, StrictRedis
import numpy as np
import msgpack_numpy as m


class LinUCB:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def create_model(self, model_name: str, actions: List[str], n_features: int, alpha: float = 0.1):
        model_meta = await self.redis.get(f"model_meta:{model_name}")
        if model_meta is not None:
            raise ValueError(f"model {model_name} already exists")
        n_actions = len(actions)
        model_meta = {
            "actions": actions,
            "n_actions": n_actions,
            "alpha": alpha,
        }
        await self.redis.set(f"model_meta:{model_name}", ujson.dumps(model_meta))
        await self.redis.set(f"action_successes:{model_name}", m.packb(np.zeros(n_actions)))
        await self.redis.set(f"action_tries:{model_name}", m.packb(np.zeros(n_actions)))
        await self.redis.set(f"covariance_matrices:{model_name}", m.packb(np.tile(
            np.identity(n_features), (n_actions, 1, 1)
        )))
        await self.redis.set(f"reward_matrix:{model_name}", m.packb(np.zeros((n_actions, n_features))))

    async def _increment_action_tries(self, model_name: str, action: str) -> None:
        model_meta = await self.redis.get(f"model_meta:{model_name}")
        actions = ujson.loads(model_meta)["actions"]
        action_tries = m.unpackb(await self.redis.get(f"action_tries:{model_name}")).copy()
        action_tries[actions.index(action)] += 1
        await self.redis.set(f"action_tries:{model_name}", m.packb(action_tries))

    async def _update_covariance_matrix(self, model_name: str, action: str, context: np.ndarray) -> None:
        model_meta = await self.redis.get(f"model_meta:{model_name}")
        actions = ujson.loads(model_meta)["actions"]
        action_index = actions.index(action)
        covariance_matrices = m.unpackb(await self.redis.get(f"covariance_matrices:{model_name}")).copy()
        covariance_matrices[action_index] = covariance_matrices[
            action_index
        ] + np.dot(context, context.T)
        await self.redis.set(f"covariance_matrices:{model_name}", m.packb(covariance_matrices))

    async def _get_action_with_max_ucb(self, model_name: str, context: np.ndarray) -> str:
        model_meta = await self.redis.get(f"model_meta:{model_name}")
        alpha = ujson.loads(model_meta)["alpha"]
        actions = ujson.loads(model_meta)["actions"]
        covariance_matrices = m.unpackb(await self.redis.get(f"covariance_matrices:{model_name}")).copy()
        reward_matrix = m.unpackb(await self.redis.get(f"reward_matrix:{model_name}")).copy()
        inverse_covariance_matrices = np.linalg.inv(covariance_matrices)
        linucb_estimates = []
        for covariance_matrix, reward_vector in zip(
            inverse_covariance_matrices, reward_matrix
        ):
            arm_coefficients = np.dot(covariance_matrix, reward_vector)
            pointwise_estimate = np.dot(arm_coefficients, context)

            upper_bound = alpha * np.sqrt(
                np.dot(np.dot(context.T, covariance_matrix), context)
            )

            linucb_estimates.append(pointwise_estimate + upper_bound)

        return actions[np.nanargmax(linucb_estimates)]

    async def select_action(self, model_name: str, context: np.ndarray) -> str:
        model_meta = await self.redis.get(f"model_meta:{model_name}")
        actions = ujson.loads(model_meta)["actions"]
        action_tries = m.unpackb(await self.redis.get(f"action_tries:{model_name}")).copy()
        untested_actions = np.nonzero(action_tries == 0)[0]
        if untested_actions.size == 0:
            best_action_so_far = self._get_action_with_max_ucb(context)
            self._increment_action_tries(model_name, best_action_so_far)
            self._update_covariance_matrix(
                model_name, best_action_so_far, context)
            return best_action_so_far
        else:
            untested_action = actions[untested_actions[0]]
            self._increment_action_tries(model_name, untested_action)
            self._update_covariance_matrix(
                model_name, untested_action, context)
            return untested_action

    async def reward_action(
        self, model_name: str, action: str, context: np.ndarray, reward: float = 1.0
    ) -> None:
        model_meta = await self.redis.get(f"model_meta:{model_name}")
        reward_matrix = m.unpackb(await self.redis.get(f"reward_matrix:{model_name}")).copy()
        actions = ujson.loads(model_meta)["actions"]
        if action not in actions:
            raise ValueError(f"action {action} not recognized")
        action_index = actions.index(action)
        reward_matrix[action_index] += reward * context
        await self.redis.set(f"reward_matrix:{model_name}", m.packb(reward_matrix))


async def main():

    redis = StrictRedis(host="127.0.0.1", port=6379, db=0)
    linucb = LinUCB(redis)
    model_name = "test"
    await linucb.create_model(model_name, ["a", "b", "c"], 3, 0.1)
    context = np.array([1, 0, 0])

    action = await linucb.select_action(model_name, context)
    print(f"{action} for context({context})")

    context = np.array([1, 1, 0])
    await linucb.reward_action(model_name, action, context, 1)

    context = np.array([0, 1, 1])
    action = await linucb.select_action(model_name, context)
    print(f"{action} for context({context})")

    context = np.array([0, 0, 1])
    action = await linucb.select_action(model_name, context)
    print(f"{action} for context({context})")
    await linucb.reward_action(model_name, action, context, 1)

    context = np.array([1, 1, 1])
    action = await linucb.select_action(model_name, context)
    print(f"{action} for context({context})")
    await linucb.reward_action(model_name, action, context, 1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
