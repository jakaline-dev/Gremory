# Gremory

An LLM inference server built with [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and [LitServe](https://github.com/Lightning-AI/LitServe).

This library aims to be an optimized inference-only version of [text-generation-ui](https://github.com/oobabooga/text-generation-webui).

Currently working on the frontend (Gremory UI), which would be a seperate library.

## Installation

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. `uv sync` will install a virtual env with libraries
3. Enter venv (`.venv\Scripts\activate` or `source .venv/bin/activate`)
4. `ltt install --pytorch-computation-backend=cu121 torch` to install torch (Please use the newest torch version, 2.4.1)
5. Rename config.yaml.example to config.yaml, add your local LLM's absolute path to model_path
6. Open server with `uv run server.py`

## How to use

`uv run server.py` will run the server on `http://localhost:9052`. It has a single endpoint (`/v1/chat/completion`) which is compatible with OpenAI specs.

To test, try out `uv run client.py`.

## Features

### OpenAI Compatible

### Sampler API

Most of the LLM inference engines have limited options on token sampling. While current LLMs work pretty well with deterministic settings (no sampling), sampling does make a big difference when applied to creative tasks such as writing, roleplay, etc.

Gremory allows users to customize the flow of the sampling process, by inputting the sampling parameters as a list.

Here's an example of a sampling parameter setting:

```json
[
    {
        "type": "min_p",
        "value": 0.1
    },
    {
        "type": "DRY",
        "multiplier": 0.85,
        "base": 1.75,
        "sequence_breakers": ["\n"],
    },
    {
        "type": "temperature",
        "value": 1.1
    },
]
```

In this configuration, the sampling process flows in the following order:
1. Min P
2. DRY
3. Temperature

#### Supported Samplers

Currently, Gremory supports the samplers listed below:
- Temperature
- Top P
- Top K
- [TFS](https://www.trentonbricken.com/Tail-Free-Sampling/)
- [Min P](https://github.com/huggingface/transformers/issues/27670)
- [DRY](https://github.com/oobabooga/text-generation-webui/pull/5677)
- [XTC](https://github.com/oobabooga/text-generation-webui/pull/6335)
- [Unified Sampler](https://docs.novelai.net/text/Editor/slidersettings.html#Unified)

You can also implement your own sampling algorithms by adding a custom [`LogitsProcessor`](https://huggingface.co/docs/transformers/internal/generation_utils#logitsprocessor) in `src/GremoryServer/modules/sampling.py`.

### Prefill Response

[What is Prefill Response?](https://docs.anthropic.com/en/api/migrating-from-text-completions-to-messages#putting-words-in-claudes-mouth)

When the last message is `assistant` role, Gremory will continue from the last message instead of adding another `assistant` message. Also, Gremory has a `add_generation_prompt` parameter, allowing to force prefill even when the last message is not `assistant`.

## TODO
- [x] Chat Template Support
- [x] Prefill Response
- [ ] GremoryUI (Currently building with Svelte 5 & shadcn-svelte)
- [ ] API wiki
- [ ] Quantized KV-Cache
- [ ] Tests