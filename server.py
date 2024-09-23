import platform
from typing import Tuple, Type

import litserve as ls
from fastapi.exceptions import HTTPException
from flask_cloudflared import start_cloudflared
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from gremory.modules.llama_cpp_wrapper import LlamaCPPWrapper
from gremory.types import GremoryRequest


class Settings(BaseSettings):
    model_path: str
    n_ctx: int
    n_gpu_layers: int = 0
    flash_attn: bool = True
    logits_all: bool = False
    verbose: bool = False
    cloudflared: bool = False

    model_config = SettingsConfigDict(yaml_file="config.yaml", protected_namespaces=())

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


settings = Settings()


class GremoryLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model = LlamaCPPWrapper(**settings.model_dump())

    def decode_request(self, request: GremoryRequest, context):
        if request.prompt and request.messages:
            raise HTTPException(400, "Cannot set both prompt and messages")
        elif request.messages:
            input = {
                "messages": request.messages,
                "functions": request.functions,
                "function_call": request.function_call,
                "tools": request.tools,
                "tool_choice": request.tool_choice,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": 0,  # Disable default
                "min_p": 1.0,  # Disable default
                "stream": request.stream,
                "stop": request.stop,
                "seed": request.seed,
                "response_format": request.response_format,
                "max_tokens": request.max_tokens,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
                "model": request.model,
                "logit_bias": request.logit_bias,
                "top_logprobs": request.top_logprobs,
                "add_generation_prompt": request.add_generation_prompt,
            }
        elif request.prompt:
            input = {
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "min_p": 1.0,  # Disable default
                "stop": request.stop,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "top_k": 0,  # Disable default
                "stream": request.stream,
                "seed": request.seed,
                "model": request.model,
                "logit_bias": request.logit_bias,
            }
        else:
            raise HTTPException(400, "Set either prompt or messages")
        if request.samplers and len(request.samplers) > 0:
            input["logits_processor"] = self.model._convert_logits_processor(
                request.samplers
            )
            input["temperature"] = 1.0 if request.do_sample else 0.0
        return input

    def predict(self, input: dict):
        if "messages" in input:
            if input["stream"]:
                for item in self.model.create_chat_completion(**input):
                    yield item
            else:
                yield self.model.create_chat_completion(**input)
        else:
            if input["stream"]:
                for item in self.model.create_completion(**input):
                    yield item
            else:
                yield self.model.create_completion(**input)

    def encode_response(self, output):
        for out in output:
            yield out


if __name__ == "__main__":
    if platform.system() == "Windows":
        try:
            import winloop

            winloop.install()
        except ImportError:
            pass
    api = GremoryLitAPI()
    server = ls.LitServer(api, stream=True, api_path="/v1/chat/completions")
    if settings.cloudflared:
        start_cloudflared(port=9052, metrics_port=8152)
    server.run(port=9052)
