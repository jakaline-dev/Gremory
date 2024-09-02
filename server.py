import os
import platform
from typing import List, Optional, Union

import litserve as ls
from dotenv import load_dotenv
from pydantic import BaseModel

from gremory.modules.llama_cpp_wrapper import LlamaCPPWrapper
from gremory.modules.sampling import Sampler

load_dotenv()


class GremoryRequest(BaseModel):
    model: Optional[str] = ""
    prompt: str
    do_sample: Optional[bool] = True
    samplers: List[Sampler] = []
    n: Optional[int] = 1
    max_tokens: Optional[int] = 100
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = True


class GremoryLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model = LlamaCPPWrapper(
            model_path=os.environ.get("MODEL_PATH"),
            n_gpu_layers=-1,
            flash_attn=True,
            logits_all=False,
            verbose=False,
        )

    def decode_request(self, request: GremoryRequest):
        if isinstance(request.samplers, list) and len(request.samplers) > 0:
            logits_processor = self.model._convert_logits_processor(request.samplers)
            do_sample = request.do_sample
        else:
            logits_processor = None
            do_sample = False
        return {
            "prompt": request.prompt,
            "logits_processor": logits_processor,
            # temperature for greedy / multinomial switch
            "temperature": 1.0 if do_sample else 0.0,
            "max_tokens": request.max_tokens,
            "stop": request.stop,
        }

    def predict(self, input):
        for item in self.model(**input, stream=self.stream):
            yield item

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
    server = ls.LitServer(api, stream=True)
    server.run(port=9052)
