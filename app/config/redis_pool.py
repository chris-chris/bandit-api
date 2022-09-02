"""Redis client module."""

from typing import AsyncIterator

from aioredis import Redis
import aioredis


async def init_redis_pool(redis_host: str, redis_port: str, redis_db: str) -> AsyncIterator[Redis]:
    redis = aioredis.Redis.from_url(
        f"redis://{redis_host}:{redis_port}/{redis_db}", 
    )
    yield redis
    await redis.close()
    await redis.connection_pool.disconnect()
