"""Containers module."""

from dependency_injector import containers, providers

import app.config.redis_pool as redis_pool
from app.algorithms.egreedy import EGreedy
from app.algorithms.ucb1 import UCB1
from app.algorithms.linucb import LinUCB


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

    uow = providers.Factory(
        src.services.unit_of_work.SqlAlchemyUnitOfWork,
        session_factory=session_factory,
    )

    v1_refresh_service = providers.Factory(
        src.routers.v1.refresh_service.V1RefreshService,
        uow=uow,
        host=config.host,
    )

    loan_event_service = providers.Factory(
        src.services.loan_event_service.LoanEventService,
        uow=uow,
        pubsub_client=pubsub_client,
    )

    sync_pine_loan_service = providers.Factory(
        src.services.sync_pine_loan_service.SyncPineLoanService,
        uow=uow,
        loan_event_service=loan_event_service,
        pine_api=pine_api,
        alchemy_api=alchemy_api,
    )

    sync_nftfi_loan_service = providers.Factory(
        src.services.sync_nftfi_loan_service.SyncNFTfiLoanService,
        uow=uow,
        loan_event_service=loan_event_service,
        nftfi_api=nftfi_api,
    )

    loan_service = providers.Factory(
        src.services.loan_service.LoanService,
        uow=uow,
        pine_api=pine_api,
        sync_pine_loan_service=sync_pine_loan_service,
        sync_nftfi_loan_service=sync_nftfi_loan_service,
    )

    loan_offer_service = providers.Factory(
        src.services.loan_offer_service.LoanOfferService,
        uow=uow,
        pine_api=pine_api,
        alchemy_api=alchemy_api,
    )
