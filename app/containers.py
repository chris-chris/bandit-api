"""Containers module."""

from dependency_injector import containers, providers

import app.config.redis_pool as redis_pool
from app.algorithms.egreedy import EGreedy
from app.algorithms.ucb1 import UCB1
from app.algorithms.linucb import LinUCB
from app.services.bandit_service import BanditService


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    ############
    # adapters
    ############

    redis_pool = providers.Resource(
        redis_pool.init_redis_pool,
        redis_host=config.redis_host,
        redis_port=config.redis_port,
        redis_db=config.redis_db,
    )

    egreedy = providers.Factory(
        EGreedy,
        redis=redis_pool,
    )

    ucb1 = providers.Factory(
        UCB1,
        redis=redis_pool,
    )

    linucb = providers.Factory(
        LinUCB,
        redis=redis_pool,
    )

    ############
    # services
    ############
    bandit_service = providers.Factory(
        BanditService,
        redis=redis_pool,
        egreedy=egreedy,
        linucb=linucb,
    )
