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
from app.containers import Container

from app.models.pymodels import BanditSelectActionRequest, GeneralResponse

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


@app.post("/v1/bandit/select-action", status_code=status.HTTP_200_OK)
async def v1_bandit_select_action(
        request: BanditSelectActionRequest,
        response: Response,
):
    try:
        if request.model_name == "baseline" and request.model_version == "1":
            data = baseline_v1(request)
        elif request.model_name == "estimate-ratio" and request.model_version == "1":
            data = estimate_ratio_v1(request)
        data, message = service.insert_strategy(request, data)

        return GeneralResponse(status_code=20000, message=message, data=data).dict(by_alias=True)
    except Exception as e:
        message = f"{e}\n{traceback.format_exc()}"
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return GeneralResponse(status_code=50000, message=f"Exception: {e}", data={}).dict(by_alias=True)


container = Container()
container.config.redis_host.from_env("REDIS_HOST", "localhost")
container.config.redis_port.from_env("REDIS_PORT", "6379")
container.config.redis_db.from_env("REDIS_DB", "0")
container.wire(modules=[
    __name__,
])
