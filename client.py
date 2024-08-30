import json

import requests

response = requests.post(
    "http://127.0.0.1:9052/predict",
    json={
        "prompt": "Kenny: Where's the ketchup? Oh,",
        "samplers": [
            {"type": "min_p", "value": 0.2},
            {
                "type": "DRY",
                "multiplier": 0.85,
                "base": 1.75,
                "sequence_breakers": ["\n"],
            },
            {"type": "temperature", "value": 1.5},
        ],
        "do_sample": True,
        "max_tokens": 400,
    },
    stream=True,
)
for line in response.iter_lines(decode_unicode=True):
    if line:
        print(json.loads(line)["choices"][0]["text"], end="")
