

import langchain_openai
from langchain_experimental.tools import PythonREPLTool

from rmrkl.agent import ChatZeroShotAgent
from rmrkl.executor import RetryAgentExecutor

# APIキーの設定
api_keys = {
    'OPENAI_API_KEY': "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
}

if not api_keys['OPENAI_API_KEY']:
    raise ValueError("OPENAI_API_KEYが環境変数に設定されていません。")

# ツールの設定
tools = [
    PythonREPLTool(),
]

# LLMの設定
llm = langchain_openai.ChatOpenAI(
    temperature=0,  # tempの代わりに0を使用
    model_name="gpt-4o-mini",  # modelの代わりに具体的なモデル名を指定
    request_timeout=1000,
    api_key=api_keys['OPENAI_API_KEY'],
)
# エージェントの作成
agent = RetryAgentExecutor.from_agent_and_tools(
    tools=tools,
    agent=ChatZeroShotAgent.from_llm_and_tools(llm, tools),
    verbose=True,
)

# エージェントの実行
result = agent.run("Who won the Nobel Peace Prize in 2023? Calculate the square of their age.")
print(result)