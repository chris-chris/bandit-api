from decimal import Decimal
from typing import List, Any, Union, Optional
from datetime import datetime

from pydantic_sqlalchemy import sqlalchemy_to_pydantic
from pydantic import BaseModel, validator, constr

from app.utils import format_datetime, format_precision


def to_camel(string: str) -> str:
    words = string.split('_')
    for i, word in enumerate(words):
        if i > 0:
            words[i] = word.capitalize()
    return ''.join(words)


class GeneralResponse(BaseModel):
    status_code: int
    message: str
    data: Any

    class Config:
        alias_generator = to_camel
        allow_population_by_field_name = True


class BanditCreateModelRequest(BaseModel):
    model_name: str
    algorithm: str
    actions: List[str]
    n_features: Optional[int]
    n_actions: int
    alpha: Optional[float] = 0.1
    epsilon: Optional[float] = 0.1


class BanditSelectActionRequest(BaseModel):
    model_name: str
    algorithm: str
    context: Optional[List[int]]

    class Config:
        alias_generator = to_camel
        allow_population_by_field_name = True
        orm_mode = True


class BanditRewardActionRequest(BaseModel):
    algorithm: str
    model_name: str
    action: str
    context: Optional[List[int]]

    class Config:
        alias_generator = to_camel
        allow_population_by_field_name = True
        orm_mode = True


class BanditRewardActionRequest(BaseModel):
    model_name: str
    algorithm: str
    action: str
    context: Optional[List[int]]

    class Config:
        alias_generator = to_camel
        allow_population_by_field_name = True
        orm_mode = True


class ContextualBanditSelectActionRequest(BaseModel):
    model_name: str
    context: List[int]

    class Config:
        alias_generator = to_camel
        allow_population_by_field_name = True
        orm_mode = True


class ContextualBanditRewardActionRequest(BaseModel):
    model_name: str
    context: List[int]
    action: str

    class Config:
        alias_generator = to_camel
        allow_population_by_field_name = True
        orm_mode = True
