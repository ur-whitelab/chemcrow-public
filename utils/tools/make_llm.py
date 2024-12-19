from langchain_openai import ChatOpenAI

def make_llm(
        model,
        temp,
        api_key
):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        llm = ChatOpenAI(
            temperature=temp,
            api_key=api_key,
            model=model,
            timeout=1000
        )
    elif model.startswith("text-"):
        llm = ChatOpenAI(
            temperature=temp,
            model=model,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Invalid model name: {model}")

    return llm

sample_llm=ChatOpenAI(
    api_key='ollama',
    model='qwen2.5:32b',
    base_url='http://192.168.31.194:8000/v1',
    temperature=0.2,
)


