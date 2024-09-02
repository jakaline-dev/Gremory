from typing import Optional

import numpy as np
import numpy.typing as npt
from llama_cpp._internals import _LlamaSamplingContext, _LlamaSamplingParams
from llama_cpp.llama import Llama
from llama_cpp.llama_grammar import LlamaGrammar
from transformers.generation.logits_process import (
    LogitsProcessorList,
    MinPLogitsWarper,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from GremoryServer.modules.sampling import (
    DRYLogitsProcessor,
    Sampler,
    TailFreeLogitsWarper,
)


class LlamaCPPWrapper(Llama):
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
                    # sampler.sequence_breakers += [
                    #     " " + s for s in sampler.sequence_breakers
                    # ]
                    # If the sequence breaker tokens are split to multiple tokens, only use their last token
                    sequence_breakers = [
                        self.tokenize(s.encode("utf-8"), add_bos=False, special=True)[
                            -1
                        ]
                        for s in sampler.sequence_breakers
                    ]
                    # Should it throw error when split to multiple tokens?
                    # assert all([len(s) == 1 for s in sequence_breakers])
                    logits_processor_list.append(
                        DRYLogitsProcessor(
                            sampler.multiplier,
                            sampler.base,
                            sampler.allowed_length,
                            sequence_breakers,
                            sampler.penalty_range,
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
        # useless
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
