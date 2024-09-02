# Gremory Server

An LLM inference server built with [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and [LitServe](https://github.com/Lightning-AI/LitServe).

Most of the ideas are from `text-generation-ui`[https://github.com/oobabooga/text-generation-webui]. This library aims to be a slimmed, optimized version of `text-generation-ui`.

Currently pre-alpha. Has a lot of stuff missing, and not OpenAI-spec compatible yet. Also working on a frontend (Gremory UI), which would be a seperate library.

## Installation & How to use

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. `uv sync` will install a virtual env with libraries
3. Enter venv (`.venv\Scripts\activate` or `source .venv/bin/activate`)
4. `ltt install --pytorch-computation-backend=cu121 torch` to install torch
5. Create a .env file
6. In the .env file, add the GGUF model file's absolute path as 'MODEL_PATH' (`MODEL_PATH=(model path)`)
7. `uv run server.py` will run the server on `http://localhost:9052`.
8. To test, try out `uv run client.py`. Customize client.py as needed.

## Sampler API

Most of the LLM inference engines have limited options on token sampling. While current LLMs work pretty well with deterministic settings (no sampling), sampling does make a big difference when applied to creative tasks such as writing, roleplay, etc.

Gremory Server allows users to customize the flow of the sampling process, by inputting the sampling parameters as a list.

### Example

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

### Supported Samplers

Currently, Gremory supports the samplers listed below:
- Temperature
- Top P
- Top K
- [TFS](https://www.trentonbricken.com/Tail-Free-Sampling/)
- [Min P](https://github.com/huggingface/transformers/issues/27670)
- [DRY](https://github.com/oobabooga/text-generation-webui/pull/5677)

You can also add your own sampling algorithms by adding a custom [LogitsProcessor](https://huggingface.co/docs/transformers/internal/generation_utils#logitsprocessor) in `src/gremory/modules/sampling.py`.

## TODO
[ ] Front-end
[ ] Make multiple endpoints (OpenAI compatible endpoint / Customizable endpoints)
[ ] Let users add their own samplers as .py files
[ ] Tests