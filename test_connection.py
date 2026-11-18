import os
from openai import OpenAI
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Initialize the client using your Experiential Labs credentials
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

print("Testing connection to the LLM...")

try:
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[
            {"role": "user", "content": "You are a performance bot. Say 'Systems ready!' in a cool way."}
        ]
    )
    print("Bot Response:", response.choices[0].message.content)
except Exception as e:
    print("Error connecting:", e)