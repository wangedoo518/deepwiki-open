"""Example script to generate embeddings using Alibaba Qwen3 model via DashScope."""

import os
from openai import OpenAI

# Initialize the OpenAI client with DashScope credentials
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

if __name__ == "__main__":
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=[
            "风急天高猿啸哀",
            "渚清沙白鸟飞回",
            "无边落木萧萧下",
            "不尽长江滚滚来",
        ],
        dimensions=1024,
        encoding_format="float",
    )
    print(response.model_dump_json())

