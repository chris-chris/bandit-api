import os

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Response,
    status,
    Request,
    Depends,
    BackgroundTasks,
)
from dependency_injector.wiring import inject, Provide

from app.algorithms.egreedy import EGreedy
from app.algorithms.linucb import LinUCB
from app.containers import Container
from app.models.pymodels import BanditCreateModelRequest, BanditRewardActionRequest, BanditSelectActionRequest, GeneralResponse
from app.services.bandit_service import BanditService

dirpath = os.path.dirname(os.path.abspath(__file__))

load_dotenv(f"{dirpath}/env/{os.environ.get('ENV', 'local')}/.env")


app = FastAPI(
    title="Bandit API",
    description="Bandit API",
    version="0.0.1",
    contact={
        "name": "Chris Hoyean Song",
        "email": "sjhshy@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


@app.post("/v1/models/create", status_code=status.HTTP_201_CREATED)
@inject
async def v1_models_create_model(
    request: BanditCreateModelRequest,
    bandit_service: BanditService = Depends(Provide[Container.bandit_service]),
):
    """
    Create a model
    """
    try:
        if request.algorithm == "egreedy":
            # await bandit_service.create_egreedy_model(request.model_name, request.actions, request.epsilon)
            await bandit_service.egreedy.create_model(request.model_name, request.actions, request.epsilon)
        elif request.algorithm == "linucb":
            await bandit_service.linucb.create_model(request.model_name, request.actions, request.n_features, request.alpha)
    except Exception as e:
        return GeneralResponse(status_code=400, message=str(e), data=None)
    return GeneralResponse(message="OK", status_code=status.HTTP_201_CREATED, data={})


@app.post("/v1/models/select-action", status_code=status.HTTP_201_CREATED)
@inject
async def select_action(
    request: BanditSelectActionRequest,
    bandit_service: BanditService = Depends(Provide[Container.bandit_service]),
):
    """
    Select an action
    """
    try:
        if request.algorithm == "egreedy":
            # await bandit_service.create_egreedy_model(request.model_name, request.actions, request.epsilon)
            action = await bandit_service.egreedy.select_action(request.model_name)
        elif request.algorithm == "linucb":
            action = await bandit_service.linucb.select_action(request.model_name, request.context)
    except Exception as e:
        return GeneralResponse(status_code=400, message=str(e), data=None)
    return GeneralResponse(message="OK", status_code=status.HTTP_201_CREATED, data={"action": action})


@app.post("/v1/models/reward-action", status_code=status.HTTP_201_CREATED)
@inject
async def reward_action(
    request: BanditRewardActionRequest,
    bandit_service: BanditService = Depends(Provide[Container.bandit_service]),
):
    """
    Select an action
    """
    try:
        if request.algorithm == "egreedy":
            # await bandit_service.create_egreedy_model(request.model_name, request.actions, request.epsilon)
            res = await bandit_service.egreedy.reward_action(request.model_name, request.action)
        elif request.algorithm == "linucb":
            res = await bandit_service.linucb.reward_action(request.model_name, request.action, request.context)
    except Exception as e:
        return GeneralResponse(status_code=400, message=str(e), data=None)
    return GeneralResponse(message="OK", status_code=status.HTTP_201_CREATED, data=res)


container = Container()
container.config.redis_host.from_env("REDIS_HOST", "localhost")
container.config.redis_port.from_env("REDIS_PORT", "6379")
container.config.redis_db.from_env("REDIS_DB", "0")
container.wire(modules=[
    __name__,
])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
