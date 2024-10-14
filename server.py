import platform
from typing import Tuple, Type, Union

import litserve as ls
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from flask_cloudflared import start_cloudflared
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from gremory.modules.llama_cpp_wrapper import LlamaCPPWrapper
from gremory.types import (
    GremoryChatCompletionsInput,
    GremoryChatCompletionsRequest,
    GremoryCompletionsInput,
    GremoryCompletionsRequest,
    LogitBiasWarper,
    TemperatureSampler,
    TopPSampler,
)


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

    def reload(self):
        del self.model
        self.model = LlamaCPPWrapper(**settings.model_dump())

    def decode_request(
        self,
        request: Union[GremoryCompletionsRequest, GremoryChatCompletionsRequest],
        context,
    ):
        if isinstance(request, GremoryChatCompletionsRequest):
            input = GremoryChatCompletionsInput(
                messages=request.messages,
                functions=request.functions,
                function_call=request.function_call,
                tools=request.tools,
                tool_choice=request.tool_choice,
                response_format=request.response_format,
                top_logprobs=request.top_logprobs,
                add_generation_prompt=request.add_generation_prompt,
                max_tokens=request.max_tokens,
                stop=request.stop,
                stream=request.stream,
                seed=request.seed,
                model=request.model,
                chat_template=request.chat_template,
            )
        elif isinstance(request, GremoryCompletionsRequest):
            input = GremoryCompletionsInput(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                stop=request.stop,
                stream=request.stream,
                seed=request.seed,
                model=request.model,
            )
        else:
            raise HTTPException(400, "Set either prompt or messages")

        logit_processor_list = []

        if request.samplers and len(request.samplers) > 0:
            logit_processor_list += request.samplers
        else:
            if request.top_p is not None:
                logit_processor_list.append(TopPSampler(value=request.top_p))
            if request.temperature is not None:
                logit_processor_list.append(
                    TemperatureSampler(value=request.temperature)
                )
            # TODO: Handle presence_penalty, frequence_penalty
        if request.logit_bias and len(request.logit_bias.items()) > 0:
            logit_processor_list.append(LogitBiasWarper(value=request.logit_bias))

        input.logits_processor_list_input = logit_processor_list
        return input

    def predict(
        self, input: Union[GremoryChatCompletionsInput, GremoryCompletionsInput]
    ):
        logits_processor = None
        if isinstance(input, GremoryChatCompletionsInput) or isinstance(
            input, GremoryCompletionsInput
        ):
            logits_processor = self.model._convert_logits_processor(
                input.logits_processor_list_input
            )
        if isinstance(input, GremoryChatCompletionsInput):
            if input.stream:
                for item in self.model.create_chat_completion(
                    **input.model_dump(exclude="logits_processor_list_input"),
                    logits_processor=logits_processor,
                    temperature=0.0,
                    top_p=0.0,
                    top_k=0.0,
                    # min_p=1.0,
                ):
                    yield item
            else:
                yield self.model.create_chat_completion(
                    **input.model_dump(exclude="logits_processor_list_input"),
                    logits_processor=logits_processor,
                    temperature=0.0,
                    top_p=0.0,
                    top_k=0.0,
                    # min_p=1.0,
                )
        elif isinstance(input, GremoryCompletionsInput):
            if input.stream:
                for item in self.model.create_completion(
                    **input.model_dump(exclude="logits_processor_list_input"),
                    logits_processor=logits_processor,
                    temperature=0.0,
                    top_p=0.0,
                    top_k=0.0,
                    # min_p=1.0,
                ):
                    yield item
            else:
                yield self.model.create_completion(
                    **input.model_dump(exclude="logits_processor_list_input"),
                    logits_processor=logits_processor,
                    temperature=0.0,
                    top_p=0.0,
                    top_k=0.0,
                    # min_p=1.0,
                )
        else:
            raise ValueError()

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
    server = ls.LitServer(
        api,
        stream=True,
        api_path="/v1/chat/completions",
        middlewares=[
            (
                CORSMiddleware,
                {
                    "allow_origins": ["*"],
                    "allow_credentials": True,
                    "allow_methods": ["*"],
                    "allow_headers": ["*"],
                },
            )
        ],
    )
    if settings.cloudflared:
        start_cloudflared(port=9052, metrics_port=8152)
    server.run(port=9052)
