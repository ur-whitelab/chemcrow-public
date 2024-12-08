import json
from typing import Any, Dict, Iterator, List, Optional

from dify_client import ChatClient
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Field, HttpUrl


class DifyCustomLLM(LLM):
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

    def _stream(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Iterator[GenerationChunk]:
        chat_response = self.chat_client.create_chat_message(inputs={}, query=prompt, user=self.user_id, response_mode="streaming")
        chat_response.raise_for_status()

        for line in chat_response.iter_lines(decode_unicode=True):
            line = line.split('data:', 1)[-1]
            if line.strip():
                line = json.loads(line.strip())
                chunk = GenerationChunk(text=line.get('answer', ''))
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": "DifyCustomLLM"}

    @property
    def _llm_type(self) -> str:
        return "dify_custom"