from typing import List

from aioredis import Redis, StrictRedis

from app.algorithms.egreedy import EGreedy
from app.algorithms.linucb import LinUCB


class BanditService:
    def __init__(self, redis: Redis, egreedy: EGreedy, linucb: LinUCB):
        self.redis = redis
        self.egreedy = egreedy
        self.linucb = linucb
