from typing import Dict, List, Literal, Optional, Union

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)
from pydantic import BaseModel

from gremory.modules.sampling import Sampler


class GremoryRequest(BaseModel):
    # OpenAI Specs
    messages: Optional[List[ChatCompletionMessageParam]] = None
    model: str = ""  #
    frequency_penalty: Optional[float] = 0.0
    function_call: Optional[completion_create_params.FunctionCall] = None
    functions: Optional[List[completion_create_params.Function]] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    presence_penalty: Optional[float] = 0.0
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
    top_p: Optional[float] = 1.0
    user: Optional[str] = None
    # Gremory
    prompt: Optional[str] = None
    do_sample: bool = True
    samplers: Optional[List[Sampler]] = None
    chat_template: Optional[Union[str, List[str]]] = None
    add_generation_prompt: bool = True
    # class Config:
    #    arbitrary_types_allowed = True
