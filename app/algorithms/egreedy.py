from typing import List
import random

import numpy as np
from aioredis import Redis, StrictRedis
import ujson
import msgpack_numpy as m


class EGreedy:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def create_model(self, model_name: str, actions: List[str], epsilon: float = 0.1):
        model_meta = await self.redis.get(f"model_meta:{model_name}")
        if model_meta is not None:
            raise ValueError(f"model {model_name} already exists")
        n_actions = len(actions)
        model_meta = {
            "actions": actions,
            "n_actions": n_actions,
            "epsilon": epsilon,
        }
        await self.redis.set(f"model_meta:{model_name}", ujson.dumps(model_meta))
        await self.redis.set(f"action_successes:{model_name}", m.packb(np.zeros(n_actions)))
        await self.redis.set(f"action_tries:{model_name}", m.packb(np.zeros(n_actions)))

    async def _increment_action_tries(self, model_name: str, action: str) -> None:
        model_meta = await self.redis.get(f"model_meta:{model_name}")
        actions = ujson.loads(model_meta)["actions"]
        action_tries = m.unpackb(await self.redis.get(f"action_tries:{model_name}")).copy()
        action_tries[actions.index(action)] += 1
        await self.redis.set(f"action_tries:{model_name}", m.packb(action_tries))

    async def _epsilon_greedy_selection(self, model_name: str):
        model_meta = await self.redis.get(f"model_meta:{model_name}")
        model_meta = ujson.loads(model_meta)
        actions = model_meta["actions"]
        epsilon = model_meta["epsilon"]
        action_tries = m.unpackb(await self.redis.get(f"action_tries:{model_name}")).copy()
        action_successes = m.unpackb(await self.redis.get(f"action_successes:{model_name}")).copy()
        if random.random() < epsilon:
            random_action = random.choice(actions)
            return random_action
        else:
            best_action_so_far = actions[
                np.nanargmax(action_successes / action_tries)
            ]
            return best_action_so_far

    async def select_action(self, model_name: str) -> str:
        model_meta = await self.redis.get(f"model_meta:{model_name}")
        model_meta = ujson.loads(model_meta)
        actions = model_meta["actions"]

        action_tries = m.unpackb(await self.redis.get(f"action_tries:{model_name}")).copy()

        untested_actions = np.nonzero(action_tries == 0)[0]
        if untested_actions.size == 0:
            epsilon_greedy_selection = self._epsilon_greedy_selection(
                model_name)
            self._increment_action_tries(model_name, epsilon_greedy_selection)
            return epsilon_greedy_selection
        else:
            untested_action = actions[untested_actions[0]]
            self._increment_action_tries(model_name, untested_action)
            return untested_action

    async def reward_action(self, model_name: str, action: str) -> None:

        model_meta = await self.redis.get(f"model_meta:{model_name}")
        model_meta = ujson.loads(model_meta)
        actions = model_meta["actions"]

        action_successes = m.unpackb(await self.redis.get(f"action_successes:{model_name}")).copy()

        if action not in actions:
            raise ValueError(f"action {action} not recognized")
        action_index = actions.index(action)
        action_successes[action_index] += 1
        await self.redis.set(f"action_successes:{model_name}", m.packb(action_successes))


async def main():
    redis = StrictRedis(host="127.0.0.1", port=6379, db=0)
    egreedy = EGreedy(redis)
    await egreedy.create_model("test", ["a", "b", "c"])
    action = await egreedy.select_action("test")
    print(f"action: {action}")
    action = await egreedy.reward_action("test", action)
    print(f"action: {action}")
    action = await egreedy.select_action("test")
    print(f"action: {action}")
    action = await egreedy.select_action("test")
    print(f"action: {action}")
    action = await egreedy.reward_action("test", action)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
