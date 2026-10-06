import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="cohere",
    api_key=os.environ["HF_TOKEN"]
)

response = client.chat.completions.create(
    model="CohereLabs/aya-vision-32b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in one sentence."},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                }
            }
        ]
    }],
    max_tokens=100
)

print(response.choices[0].message.content)