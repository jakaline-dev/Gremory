import json

import requests

response = requests.post(
    "http://127.0.0.1:9052/predict",
    json={
        "prompt": "Kenny: Where's the ketchup? Oh,",
        "samplers": [
            {
                "type": "DRY",
                "multiplier": 0.85,
                "base": 1.75,
                "sequence_breakers": ["\n"],
            },
            {"type": "XTC", "threshold": 0.1, "probability": 0.5},
            # {"type": "min_p", "value": 0.1},
            {"type": "temperature", "value": 1.1},
        ],
        "do_sample": True,
        "max_tokens": 400,
    },
    stream=True,
)
for line in response.iter_lines(decode_unicode=True):
    if line:
        print(json.loads(line)["choices"][0]["text"], end="")
