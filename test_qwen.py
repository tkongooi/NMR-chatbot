from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model_name="qwen-plus",  # Or another Qwen model like "qwen-plus"
# or choose "qwen-plus" or "qwen-turbo-latest" or qwen-max (maybe not free),  # Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# The usage is identical
response = llm.invoke("Hello, who are you? Tell me a joke")
print(response.content)
