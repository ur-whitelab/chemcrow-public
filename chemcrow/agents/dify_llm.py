import json
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

from dify_client import ChatClient
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, HttpUrl


class DifyCustomLLM(BaseChatModel):
    api_key: str = Field(..., description="API key for Dify")
    user_id: str = Field(..., description="User ID for Dify")
    base_url: HttpUrl = Field(..., description="Base URL for Dify API")
    chat_client: ChatClient = Field(None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.chat_client = ChatClient(api_key=self.api_key)
        self.chat_client.base_url = str(self.base_url)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        chat_response = self.chat_client.create_chat_message(inputs={}, query=prompt, user=self.user_id, response_mode="blocking")
        chat_response.raise_for_status()
        response_data = chat_response.json()
        response_text = response_data.get('answer', '')
        
        # stopワードが指定されている場合、応答テキストを適切に処理
        if stop:
            for stop_word in stop:
                if stop_word in response_text:
                    response_text = response_text[:response_text.index(stop_word)]
        
        return response_text
        

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        additional_kwargs = {}  
        if "tools" in kwargs:
            tools_description = self._render_tools_description(kwargs["tools"])
            prompt = f"system: {tools_description}\n{prompt}"
            kwargs.pop("tools")
        
        if "tool_choice" in kwargs:
            kwargs.pop("tool_choice")

        chat_response = self.chat_client.create_chat_message(
            inputs={},
            query=prompt,
            user=self.user_id,
            response_mode="streaming"
        )
        chat_response.raise_for_status()

        accumulated_text = ""
        for line in chat_response.iter_lines(decode_unicode=True):
            line = line.split('data:', 1)[-1]
            if line.strip():
                line = json.loads(line.strip())
                text = line.get('answer', '')
                accumulated_text += text

                # stopワードのチェック
                if stop:
                    for stop_word in stop:
                        if stop_word in text:
                            text = text[:text.index(stop_word)]
                            break

                # AIMessageを使用してChatGenerationChunkを作成
                message = AIMessageChunk(content=text)
                chunk = ChatGenerationChunk(message=message)
                if run_manager:
                    run_manager.on_llm_new_token(text, chunk=chunk)
                yield chunk


        # ストリーミング完了後、ツール呼び出しの処理
        if "tools" in kwargs:
            try:
                response_json = json.loads(accumulated_text)
                if "name" in response_json and "arguments" in response_json:
                    tool_call = {
                        "id": f"call_{uuid.uuid4().hex}",
                        "function": {
                            "arguments": json.dumps(response_json["arguments"]),
                            "name": response_json["name"],
                        },
                        "type": "function",
                    }
                    message = AIMessageChunk(content="", additional_kwargs={"tool_calls": [tool_call]})
                    chunk = ChatGenerationChunk(message=message)
                    yield chunk
            except json.JSONDecodeError:
                pass
    
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
        additional_kwargs = {}
        did_bind_tools = False
        if "tools" in kwargs:
        # ツールが提供されている場合、システムプロンプトを追加
            tools_description = self._render_tools_description(kwargs["tools"])
            messages.insert(0, SystemMessage(content=tools_description))
            did_bind_tools = True
            kwargs.pop("tools")  # toolsパラメータを削除
    
        if "tool_choice" in kwargs:
            kwargs.pop("tool_choice")

        chat_response = self.chat_client.create_chat_message(
            inputs={},
            query=prompt,
            user=self.user_id,
            response_mode="blocking"
        )
        chat_response.raise_for_status()
        response_data = chat_response.json()
        response_text = response_data.get('answer', '')

        # stopワードの処理
        if stop:
            for stop_word in stop:
                if stop_word in response_text:
                    response_text = response_text[:response_text.index(stop_word)]
        if did_bind_tools:
            try:
                response_json = json.loads(response_text)
                if "name" in response_json and "arguments" in response_json:
                    tool_call = {
                        "id": f"call_{uuid.uuid4().hex}",
                        "function": {
                            "arguments": json.dumps(response_json["arguments"]),
                            "name": response_json["name"],
                        },
                        "type": "function",
                    }
                    additional_kwargs["tool_calls"] = [tool_call]
                    response_text = ""  # OpenAIのスタイルに合わせて空にする
            except json.JSONDecodeError:
                pass
        # ChatGenerationとChatResultの作成
        generation = ChatGeneration(
        message=AIMessage(
            content=response_text,
            additional_kwargs=additional_kwargs
        )
    )
        return ChatResult(generations=[generation])

    def _render_tools_description(self, tools: List[Any]) -> str:
        """ツールの説明をシステムプロンプト用にレンダリング"""
        tool_descriptions = []
        for tool in tools:
            if isinstance(tool, dict):
                desc = f"名前: {tool.get('name')}\n説明: {tool.get('description')}"
                tool_descriptions.append(desc)
    
        return "利用可能なツール:\n" + "\n\n".join(tool_descriptions)
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """ツールをLLMにバインドするメソッド。

        Args:
            tools: バインドするツール定義のリスト。
            tool_choice: 使用するツールの指定方法。
            **kwargs: 追加のパラメータ
        """
        # ツールをOpenAI形式に変換
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        # tool_choiceの検証と処理
        if tool_choice is not None and tool_choice:
            if isinstance(tool_choice, str) and tool_choice not in ("auto", "any", "none"):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, dict) and len(formatted_tools) != 1:
                raise ValueError(
                    "tool_choiceを指定する場合は、ちょうど1つのツールを提供する必要があります。"
                    f"現在のツール数: {len(formatted_tools)}"
                )
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError(
                        "tool_choiceをTrueにできるのは1つのツールがある場合のみです。"
                        f"現在のツール数: {len(tools)}"
                    )
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

        return super().bind(tools=tools, **kwargs)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": "DifyCustomLLM"}

    @property
    def _llm_type(self) -> str:
        return "dify_custom"