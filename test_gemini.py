import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv() # This line loads the variables from your .env file

gemini_api_key = os.getenv("GOOGLE_API_KEY")
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")


from langchain_google_genai import ChatGoogleGenerativeAI

# The API key is automatically read from the GOOGLE_API_KEY environment variable
#llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

# Much cheaper to use 2.5 flash
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#cheapest to use 2.5 lite,
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

# You can now use this 'llm' object in your RAG chain
# For example:
response = llm.invoke("Hello, who are you? Tell me a joke")
print(response.content)


genai.configure(api_key=gemini_api_key)
print("Available models that support 'generateContent':")
print("-" * 45)

for m in genai.list_models():
  # The error mentioned 'generateContent', so let's check for models that support it
  if 'generateContent' in m.supported_generation_methods:
    print(f"Model name: {m.name}")
