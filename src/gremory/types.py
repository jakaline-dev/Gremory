from typing import Dict, List, Literal, Optional, Union

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)
from pydantic import BaseModel


class BaseSampler(BaseModel):
    pass


class TemperatureSampler(BaseSampler):
    value: float
    type: Literal["temperature"] = "temperature"


class TopPSampler(BaseSampler):
    value: float
    type: Literal["top_p"] = "top_p"


class TopKSampler(BaseSampler):
    value: float
    type: Literal["top_k"] = "top_k"


class MinPSampler(BaseSampler):
    value: float
    type: Literal["min_p"] = "min_p"


class TFSSampler(BaseSampler):
    value: float
    type: Literal["tfs"] = "tfs"


class DRYSampler(BaseSampler):
    multiplier: Optional[float] = 0.0
    base: Optional[float] = 1.75
    allowed_length: Optional[int] = 2
    sequence_breakers: Optional[list[str]] = []
    penalty_range: Optional[int] = 0
    type: Literal["DRY"] = "DRY"


class XTCSampler(BaseSampler):
    threshold: Optional[float] = 0.1
    probability: Optional[float] = 0.0
    type: Literal["XTC"] = "XTC"


class UnifiedSampler(BaseSampler):
    linear: Optional[float] = 0.3
    conf: Optional[float] = 0.0
    quad: Optional[float] = 0.19
    type: Literal["unified"] = "unified"


class LogitBiasWarper:
    value: list[dict[str, int]]
    type: Literal["logit_bias"] = "logit_bias"


Sampler = Union[
    TemperatureSampler
    | TopPSampler
    | TopKSampler
    | MinPSampler
    | TFSSampler
    | DRYSampler
    | XTCSampler
    | UnifiedSampler
]


class GremoryRequest(BaseModel):
    # OpenAI Specs
    messages: Optional[List[ChatCompletionMessageParam]] = None
    model: str = ""  #
    frequency_penalty: Optional[float] = None
    function_call: Optional[completion_create_params.FunctionCall] = None
    functions: Optional[List[completion_create_params.Function]] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[completion_create_params.ResponseFormat] = None
    seed: Optional[int] = None
    service_tier: Optional[Literal["auto", "default"]] = None
    stop: Union[Optional[str], List[str]] = None
    stream: Optional[bool] = False
    stream_options: Optional[ChatCompletionStreamOptionsParam] = None
    temperature: Optional[float] = None
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None
    tools: Optional[List[ChatCompletionToolParam]] = None
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    # Gremory
    prompt: Optional[str] = None
    samplers: Optional[List[Sampler]] = None
    add_generation_prompt: bool = True
