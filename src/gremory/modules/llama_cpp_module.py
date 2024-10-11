import importlib
import platform
from typing import Sequence

import numpy as np
from tqdm import tqdm

imported_module = None


def llama_cpp_lib():
    global imported_module

    lib_names = ["llama_cpp_cuda", "llama_cpp"]

    for lib_name in lib_names:
        try:
            return_lib = importlib.import_module(lib_name)
            imported_module = lib_name
            # monkey_patch_llama_cpp_python(return_lib)
            return return_lib
        except ImportError:
            continue
    print("None?")
    return None


def eval_with_progress(self, tokens: Sequence[int]):
    """
    A copy of

    https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py

    with tqdm to show prompt processing progress.
    """
    self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)

    if len(tokens) > self.n_batch:
        progress_bar = tqdm(
            range(0, len(tokens), self.n_batch), desc="Prompt evaluation", leave=False
        )
    else:
        progress_bar = range(0, len(tokens), self.n_batch)

    for i in progress_bar:
        batch = tokens[i : min(len(tokens), i + self.n_batch)]
        n_past = self.n_tokens
        n_tokens = len(batch)
        self._batch.set_batch(
            batch=batch, n_past=n_past, logits_all=self.context_params.logits_all
        )
        self._ctx.decode(self._batch)
        # Save tokens
        self.input_ids[n_past : n_past + n_tokens] = batch
        # Save logits
        if self.context_params.logits_all:
            rows = n_tokens
            cols = self._n_vocab
            logits = np.ctypeslib.as_array(self._ctx.get_logits(), shape=(rows * cols,))
            self.scores[n_past : n_past + n_tokens, :].reshape(-1)[::] = logits
            self.last_updated_index = n_past + n_tokens - 1
        else:
            rows = 1
            cols = self._n_vocab
            logits = np.ctypeslib.as_array(self._ctx.get_logits(), shape=(rows * cols,))
            last_token_index = min(n_past + n_tokens - 1, self.scores.shape[0] - 1)
            self.scores[last_token_index, :] = logits.reshape(-1)
            self.last_updated_index = last_token_index
        # Update n_tokens
        self.n_tokens += n_tokens


def monkey_patch_llama_cpp_python(lib):
    if getattr(lib.Llama, "_is_patched", False):
        # If the patch is already applied, do nothing
        return

    # lib.Llama.eval = eval_with_progress
    # lib.Llama.original_generate = lib.Llama.generate
    # lib.Llama.generate = my_generate

    # Set the flag to indicate that the patch has been applied
    lib.Llama._is_patched = True
