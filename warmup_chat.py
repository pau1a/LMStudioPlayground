from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
#MODEL = "llama3-8b"  # or whichever chat model you listed earlier
MODEL = "mixtral-latest"

intro = """
You’re an AI researcher’s assistant. Keep replies short, sharp, and accurate.
Answer in 2-3 sentences maximum.
"""

while True:
    prompt = input("\nYou > ").strip()
    if prompt.lower() in {"exit", "quit"}:
        break

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": intro},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=250,
    )
    print("AI >", resp.choices[0].message.content)
