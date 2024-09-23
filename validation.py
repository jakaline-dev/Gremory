from gremory.types import GremoryRequest

GremoryRequest.model_validate(
    {
        "messages": [
            {"role": "user", "content": "What is your favourite condiment?"},
            {
                "role": "assistant",
                "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
            },
            {"role": "user", "content": "Do you have mayonnaise recipes?"},
            # {
            #     "role": "assistant",
            #     "content": "No, but I",
            # },
        ],
        # "prompt": "What is your favourite condiment? I"
        "samplers": [
            {
                "type": "DRY",
                "multiplier": 0.85,
                "base": 1.75,
                "allowed_length": 2,
                "sequence_breakers": ["\n"],
            },
            {"type": "XTC", "threshold": 0.1, "probability": 0.5},
            # {"type": "min_p", "value": 0.1},
            {"type": "temperature", "value": 1.1},
        ],
        "do_sample": True,
        "max_tokens": 400,
        "stream": True,
    }
)
