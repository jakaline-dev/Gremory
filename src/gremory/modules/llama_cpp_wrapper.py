import ctypes
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import torch
from llama_cpp_cuda import (
    LogitsProcessorList,
    llama_token_data,
)
from llama_cpp_cuda._internals import LlamaSampler
from llama_cpp_cuda.llama_grammar import LlamaGrammar
from llama_cpp_cuda.llama_types import (
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
from gremory.modules.llama_cpp_module import llama_cpp_lib
from gremory.modules.samplers import (
    DRYLogitsProcessor,
    TailFreeLogitsWarper,
    UnifiedLogitsWarper,
    XTCLogitsWarper,
)
from gremory.types import LogitProcessorListInputType


class LlamaCPPWrapper(llama_cpp_lib().Llama):
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

    def _convert_logits_processor(self, logit_processors: LogitProcessorListInputType):
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
                case "unified":
                    logits_processor_list.append(
                        UnifiedLogitsWarper(
                            linear=logit_processor.linear,
                            conf=logit_processor.conf,
                            quad=logit_processor.quad,
                        )
                    )
                case "logit_bias":
                    logits_processor_list.append(
                        SequenceBiasLogitsProcessor(sequence_bias=logit_processor.value)
                    )
                case _:
                    raise ValueError("Undefined sampler name detected!")
        return LogitsProcessorList(logits_processor_list)

    def _init_sampler(
        self,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        **kwargs,
    ):
        sampler = LlamaSampler()

        if logits_processor is not None:
            # Create and add a custom sampler
            def apply_func(token_data_array):
                size = token_data_array.contents.size
                data_soa = token_data_array.contents.data
                data_soa_address = ctypes.addressof(data_soa.contents)

                recarray = np.recarray(
                    shape=(size,),
                    dtype=np.dtype(
                        [("id", np.intc), ("logit", np.single), ("p", np.single)],
                        align=True,
                    ),
                    buf=(llama_token_data * size).from_address(data_soa_address),
                )

                # Convert input_ids to PyTorch tensor
                input_ids_tensor = torch.tensor(
                    self._input_ids, dtype=torch.long
                ).unsqueeze(0)

                for logit_processor in logits_processor:
                    # Convert logits to PyTorch tensor
                    logits_tensor = torch.from_numpy(recarray.logit).unsqueeze(0)

                    # Apply logit processor
                    processed_logits = logit_processor(input_ids_tensor, logits_tensor)

                    # Update recarray with processed logits
                    recarray.logit[:] = processed_logits.squeeze(0).numpy()

                return recarray

            sampler.add_custom(apply_func)

        # sampler.add_penalties(
        #     n_vocab=self._n_vocab,
        #     special_eos_id=self._token_eos,
        #     linefeed_id=self._token_nl,
        #     penalty_last_n=self.last_n_tokens_size,
        #     penalty_repeat=repeat_penalty,
        #     penalty_freq=frequency_penalty,
        #     penalty_present=presence_penalty,
        #     penalize_nl=penalize_nl,
        #     ignore_eos=False,
        # )

        if grammar is not None:
            sampler.add_grammar(self._model, grammar)

        sampler.add_softmax()
        sampler.add_dist(self._seed)
        #    sampler.add_greedy()
        return sampler
