import json

import requests

STREAM = True
REQUEST_BODY = {
    "messages": [
        {"role": "user", "content": "What is your favourite condiment?"},
        {
            "role": "assistant",
            "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
        },
        {
            "role": "user",
            "content": "Do you have mayonnaise recipes?",
        },
        {
            "role": "assistant",
            "content": "No, but I can",
        },
    ],
    # "prompt": "What is your favourite condiment? I"
    "samplers": [
        # {
        #     "type": "DRY",
        #     "multiplier": 0.85,
        #     "base": 1.75,
        #     "allowed_length": 2,
        #     "sequence_breakers": ["\n"],
        # },
        {"type": "XTC", "threshold": 0.1, "probability": 0.5},
        # {"type": "min_p", "value": 0.1},
        # {"type": "temperature", "value": 1.25},
        # {"type": "unified", "linear": 0.3, "conf": 0.0, "quad": 0.19}
    ],
    "max_tokens": 200,
    "stream": STREAM,
    # "add_generation_prompt": False,
}

response = requests.post(
    "http://127.0.0.1:9052/v1/chat/completions",
    json=REQUEST_BODY,
    stream=STREAM,
)
for line in response.iter_lines(decode_unicode=True):
    if line:
        try:
            chunk = json.loads(line.decode("utf-8").strip())
            if "choices" in chunk and len(chunk["choices"]) > 0:
                chunk_0 = chunk["choices"][0]
                if "delta" in chunk_0 and "content" in chunk_0["delta"]:
                    print(chunk_0["delta"]["content"], end="", flush=True)
                elif "text" in chunk_0:
                    print(chunk_0["text"], end="", flush=True)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON: {line}")
        except Exception as e:
            print(f"Error processing chunk: {e}")

print()
