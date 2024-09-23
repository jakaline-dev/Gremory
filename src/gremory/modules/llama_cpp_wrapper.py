from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import numpy.typing as npt
from llama_cpp._internals import _LlamaSamplingContext, _LlamaSamplingParams
from llama_cpp.llama import Llama
from llama_cpp.llama_grammar import LlamaGrammar
from llama_cpp.llama_types import (
    ChatCompletionFunction,
    ChatCompletionRequestFunctionCall,
    ChatCompletionRequestMessage,
    ChatCompletionRequestResponseFormat,
    ChatCompletionTool,
    ChatCompletionToolChoiceOption,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    MinPLogitsWarper,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from gremory.modules.chat_formatter import GremoryJinja2ChatFormatter
from gremory.modules.sampling import (
    DRYLogitsProcessor,
    Sampler,
    TailFreeLogitsWarper,
    XTCLogitsWarper,
)


class LlamaCPPWrapper(Llama):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        template_choices = dict(
            (name[10:], template)
            for name, template in self.metadata.items()
            if name.startswith("tokenizer.chat_template.")
        )

        if "tokenizer.chat_template" in self.metadata:
            template_choices["chat_template.default"] = self.metadata[
                "tokenizer.chat_template"
            ]
        chat_template = template_choices[self.chat_format]
        eos_token_id = self.token_eos()
        bos_token_id = self.token_bos()

        eos_token = (
            self._model.token_get_text(eos_token_id) if eos_token_id != -1 else ""
        )
        bos_token = (
            self._model.token_get_text(bos_token_id) if bos_token_id != -1 else ""
        )
        # Monkey-patch chat handler
        self.chat_handler = GremoryJinja2ChatFormatter(
            template=chat_template,
            eos_token=eos_token,
            bos_token=bos_token,
            stop_token_ids=[eos_token],
        ).to_chat_handler()

    def create_chat_completion(
        self,
        messages: List[ChatCompletionRequestMessage],
        functions: Optional[List[ChatCompletionFunction]] = None,
        function_call: Optional[ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[ChatCompletionTool]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[ChatCompletionRequestResponseFormat] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        add_generation_prompt: bool = True,
    ) -> Union[
        CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]
    ]:
        return self.chat_handler(
            llama=self,
            messages=messages,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            stream=stream,
            stop=stop,
            seed=seed,
            response_format=response_format,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
            add_generation_prompt=add_generation_prompt,
        )

    def _convert_logits_processor(self, samplers: list[Sampler]):
        logits_processor_list = []
        for sampler in samplers:
            match sampler.type:
                case "temperature":
                    logits_processor_list.append(TemperatureLogitsWarper(sampler.value))
                case "top_p":
                    logits_processor_list.append(TopPLogitsWarper(sampler.value))
                case "top_k":
                    logits_processor_list.append(TopKLogitsWarper(sampler.value))
                case "min_p":
                    logits_processor_list.append(MinPLogitsWarper(sampler.value))
                case "tfs":
                    logits_processor_list.append(TailFreeLogitsWarper(sampler.value))
                case "DRY":
                    # Most tokenizers have different tokens when not start of a sentence ("a" and " a")
                    sampler.sequence_breakers += [
                        " " + s for s in sampler.sequence_breakers
                    ]
                    # If the sequence breaker tokens are split to multiple tokens, only use their last token
                    sequence_breakers = set()
                    for s in sampler.sequence_breakers:
                        sequence_breakers.add(
                            self.tokenize(
                                s.encode("utf-8"), add_bos=False, special=False
                            )[-1]
                        )
                    # Should it throw error when split to multiple tokens?
                    # assert all([len(s) == 1 for s in sequence_breakers])
                    logits_processor_list.append(
                        DRYLogitsProcessor(
                            multiplier=sampler.multiplier,
                            base=sampler.base,
                            allowed_length=sampler.allowed_length,
                            sequence_breakers=sequence_breakers,
                            _range=sampler.penalty_range,
                        )
                    )
                case "XTC":
                    special_token_ids = set()
                    special_token_ids.add(
                        self.tokenize(
                            "\n".encode("utf-8"), add_bos=False, special=False
                        )[0]
                    )
                    if self.token_eos():
                        special_token_ids.add(self.token_eos())
                    logits_processor_list.append(
                        XTCLogitsWarper(
                            threshold=sampler.threshold,
                            probability=sampler.probability,
                            special_token_ids=list(special_token_ids),
                        )
                    )
                case _:
                    raise ValueError("Undefined sampler name detected!")

    def sample(
        self,
        logits_processor: Optional[LogitsProcessorList] = None,
        temp: float = 1.0,
        idx: Optional[int] = None,
        grammar: Optional[LlamaGrammar] = None,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_eta: float = 0.1,
        mirostat_tau: float = 5.0,
        penalize_nl: bool = True,
    ):
        assert self._ctx is not None
        assert self.n_tokens > 0

        if idx is None:
            logits: npt.NDArray[np.single] = self._scores[-1, :]
        else:
            logits = self._scores[idx, :]

        # If logits_processor is not None, we reroute the default sampling code
        if logits_processor is not None:
            logits[:] = (
                logits_processor(self._input_ids, logits)
                if idx is None
                else logits_processor(self._input_ids[: idx + 1], logits)
            )
            # softmax
            if temp == 1.0:
                probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
                output = np.random.choice(probs.shape[-1], size=1, p=probs.ravel())
            else:
                output = np.argmax(probs, axis=-1)[np.newaxis, :]
            return output[0]
        else:
            sampling_params = _LlamaSamplingParams(
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                tfs_z=tfs_z,
                typical_p=typical_p,
                temp=temp,
                penalty_last_n=self.last_n_tokens_size,
                penalty_repeat=repeat_penalty,
                penalty_freq=frequency_penalty,
                penalty_present=presence_penalty,
                mirostat=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                penalize_nl=penalize_nl,
            )
            sampling_context = _LlamaSamplingContext(
                params=sampling_params,
                grammar=grammar,
            )
            sampling_context.prev = list(self.eval_tokens)
            id = sampling_context.sample(ctx_main=self._ctx, logits_array=logits)
            sampling_context.accept(
                ctx_main=self._ctx,
                id=id,
                apply_grammar=grammar is not None,
            )
            return id
