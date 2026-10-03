import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

chat = client.chats.create(
    model="gemini-3.8-flash",
    config=genai.types.GenerateContentConfig(
        system_instruction="You are a helpful, friendly AI assistant built to power a mobile app."
    )
)

print("=== AI Chatbot Active (Type 'exit' to quit) ===\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    try:
        response = chat.send_message(user_input)
        print(f"\nAI: {response.text}\n")
    except Exception as e:
        print(f"\n[Server Busy / Error]: {e}\nPlease try sending your message again.\n")