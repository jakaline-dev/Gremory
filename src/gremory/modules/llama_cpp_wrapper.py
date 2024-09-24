import inspect
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
from llama_cpp.llama import Llama, LogitsProcessorList
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
    MinPLogitsWarper,
    SequenceBiasLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from gremory.modules.chat_formatter import GremoryJinja2ChatFormatter
from gremory.modules.samplers import (
    DRYLogitsProcessor,
    TailFreeLogitsWarper,
    XTCLogitsWarper,
)
from gremory.types import LogitBiasWarper, Sampler


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

    def _convert_logits_processor(
        self, logit_processors: list[Union[Sampler, LogitBiasWarper]]
    ):
        logits_processor_list = []
        for logit_processor in logit_processors:
            match logit_processor.type:
                case "temperature":
                    logits_processor_list.append(
                        TemperatureLogitsWarper(logit_processor.value)
                    )
                case "top_p":
                    logits_processor_list.append(
                        TopPLogitsWarper(logit_processor.value)
                    )
                case "top_k":
                    logits_processor_list.append(
                        TopKLogitsWarper(logit_processor.value)
                    )
                case "min_p":
                    logits_processor_list.append(
                        MinPLogitsWarper(logit_processor.value)
                    )
                case "tfs":
                    logits_processor_list.append(
                        TailFreeLogitsWarper(logit_processor.value)
                    )
                case "DRY":
                    # Most tokenizers have different tokens when not start of a sentence ("a" and " a")
                    logit_processor.sequence_breakers += [
                        " " + s for s in logit_processor.sequence_breakers
                    ]
                    # If the sequence breaker tokens are split to multiple tokens, only use their last token
                    sequence_breakers = set()
                    for s in logit_processor.sequence_breakers:
                        sequence_breakers.add(
                            self.tokenize(
                                s.encode("utf-8"), add_bos=False, special=False
                            )[-1]
                        )
                    # Should it throw error when split to multiple tokens?
                    # assert all([len(s) == 1 for s in sequence_breakers])
                    logits_processor_list.append(
                        DRYLogitsProcessor(
                            multiplier=logit_processor.multiplier,
                            base=logit_processor.base,
                            allowed_length=logit_processor.allowed_length,
                            sequence_breakers=sequence_breakers,
                            _range=logit_processor.penalty_range,
                        )
                    )
                case "XTC":
                    special_token_ids = set()
                    special_token_ids.add(
                        self.tokenize(
                            "\n".encode("utf-8"), add_bos=False, special=False
                        )[-1]
                    )
                    if self.token_eos():
                        special_token_ids.add(self.token_eos())
                    logits_processor_list.append(
                        XTCLogitsWarper(
                            threshold=logit_processor.threshold,
                            probability=logit_processor.probability,
                            special_token_ids=list(special_token_ids),
                        )
                    )
                case "logit_bias":
                    logits_processor_list.append(
                        SequenceBiasLogitsProcessor(sequence_bias=logit_processor.value)
                    )
                case _:
                    raise ValueError("Undefined sampler name detected!")
        return LogitsProcessorList(logits_processor_list)

    def sample(
        self,
        logits_processor: Optional[LogitsProcessorList] = None,
        temp: float = None,
        idx: Optional[int] = None,
        grammar: Optional[LlamaGrammar] = None,
        top_k: int = None,
        top_p: float = None,
        min_p: float = None,
        typical_p: float = None,
        repeat_penalty: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        tfs_z: float = None,
        mirostat_mode: int = None,
        mirostat_eta: float = None,
        mirostat_tau: float = None,
        penalize_nl: bool = False,
    ):
        assert self._ctx is not None
        assert self.n_tokens > 0

        if idx is None:
            logits: npt.NDArray[np.single] = self._scores[-1, :]
        else:
            logits = self._scores[idx, :]

        if logits_processor is not None:
            logits[:] = (
                logits_processor(
                    torch.from_numpy(self._input_ids).unsqueeze(dim=0),
                    torch.from_numpy(logits).unsqueeze(dim=0),
                )
                if idx is None
                else logits_processor(
                    torch.from_numpy(self._input_ids[: idx + 1]).unsqueeze(dim=0),
                    torch.from_numpy(logits).unsqueeze(dim=0),
                )
            )
        # TODO: Check with transformers dimensions
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        # softmax
        output = np.random.choice(probs.shape[-1], size=1, p=probs.ravel())
        return output[0]


# sampling_params = _LlamaSamplingParams(
#     top_k=top_k,
#     top_p=top_p,
#     min_p=min_p,
#     tfs_z=tfs_z,
#     typical_p=typical_p,
#     temp=temp,
#     penalty_last_n=self.last_n_tokens_size,
#     penalty_repeat=repeat_penalty,
#     penalty_freq=frequency_penalty,
#     penalty_present=presence_penalty,
#     mirostat=mirostat_mode,
#     mirostat_tau=mirostat_tau,
#     mirostat_eta=mirostat_eta,
#     penalize_nl=penalize_nl,
# )
# sampling_context = _LlamaSamplingContext(
#     params=sampling_params,
#     grammar=grammar,
# )
# sampling_context.prev = list(self.eval_tokens)
# id = sampling_context.sample(ctx_main=self._ctx, logits_array=logits)
# sampling_context.accept(
#     ctx_main=self._ctx,
#     id=id,
#     apply_grammar=grammar is not None,
# )
# return id
