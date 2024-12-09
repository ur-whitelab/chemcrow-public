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
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
)
from langchain_core.output_parsers import JsonOutputParser
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
        
        # Process the response text appropriately if stop words are specified
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

                # Check for stop words
                if stop:
                    for stop_word in stop:
                        if stop_word in text:
                            text = text[:text.index(stop_word)]
                            break

                # Create ChatGenerationChunk using AIMessage
                message = AIMessageChunk(content=text)
                chunk = ChatGenerationChunk(message=message)
                if run_manager:
                    run_manager.on_llm_new_token(text, chunk=chunk)
                yield chunk


        # Process tool calls after streaming is complete
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
        additional_kwargs = {}
        did_bind_tools = False
        if "tools" in kwargs:
            
        # Add system prompt if tools are provided
            prompt_template = """You are an AI assistant that carefully follows this process:

1. THOUGHT:
Analyze the previous response (if any) and the user's question:
- What information was obtained from the previous tool use
- What is being asked in the new question
- What additional information might be needed

2. REASONING:
Based on your analysis, determine:
- Whether you need to use a tool
- Which tool would be most appropriate
- What arguments the tool needs

3. ACTION:
IMPORTANT: When using a tool, you must ALWAYS respond with a JSON object and NOTHING ELSE.
Be sure to enclose the code in code blocks beginning with ```json'''!
DO NOT include any explanatory text or additional content around the JSON.
The JSON must follow this exact format:
{
    "name": "tool_name",
    "arguments": {
        "arg1": "value1",
        "arg2": "value2"
    }
}
You can use the following tools:
{tools_description goes here}

- If no tool is needed, proceed to the final answer step

4. REFLECTION:
After receiving tool results:
- Evaluate if the results answer the original question
- Determine if additional tool calls are needed
- Consider if you have enough information for a final answer

5. FINAL ANSWER:
When you have sufficient information:
- Provide a clear, direct answer to the user's question
- Reference the information obtained from tools
- Explain your reasoning if appropriate

Always structure your response as:
THOUGHT: [your analysis]
REASONING: [your reasoning process]
ACTION: [tool JSON or "Proceed to final answer"]
[If tool was used] REFLECTION: [your reflection on results]
[When ready] FINAL ANSWER: [your response to the user]
                """
            tools_description =prompt_template.replace("{tools_description goes here}", self._render_tools_description(kwargs["tools"])) 
            messages.insert(0, SystemMessage(content=tools_description))
            did_bind_tools = True
            kwargs.pop("tools")  # Remove tools parameter
    
        if "tool_choice" in kwargs:
            kwargs.pop("tool_choice")
        prompt = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
        chat_response = self.chat_client.create_chat_message(
            inputs={},
            query=prompt,
            user=self.user_id,
            response_mode="blocking"
        )
        chat_response.raise_for_status()
        response_data = chat_response.json()
        response_text = response_data.get('answer', '')

        # Process stop words
        if stop:
            for stop_word in stop:
                if stop_word in response_text:
                    response_text = response_text[:response_text.index(stop_word)]
        if did_bind_tools:
            try:
                response_json = JsonOutputParser().parse(response_text)
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
                    response_text = ""  # Set to empty to match OpenAI style
            except (json.JSONDecodeError, OutputParserException):
            # JSONパースに失敗した場合は通常のテキスト応答として処理
                pass
        # Create ChatGeneration and ChatResult
        generation = ChatGeneration(
        message=AIMessage(
            content=response_text,
            additional_kwargs=additional_kwargs
        )
    )
        return ChatResult(generations=[generation])

    def _render_tools_description(self, tools: List[Any]) -> str:
        """Render tool descriptions for system prompt"""
        tool_descriptions = []
        for tool in tools:
            if isinstance(tool, dict):
                desc = f"Name: {tool.get('name')}\nDescription: {tool.get('description')}"
            elif isinstance(tool, BaseTool):
                desc = f"Name: {tool.name}\nDescription: {tool.description}\nArguments: {tool.args}" 
            else:
                desc = f"Name: {tool.__class__.__name__}\nDescription: {str(tool)}"
            tool_descriptions.append(desc)
    
        return "Available tools:\n" + "\n\n".join(tool_descriptions)
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Method to bind tools to the LLM.

        Args:
            tools: List of tool definitions to bind.
            tool_choice: Method to specify which tool to use.
            **kwargs: Additional parameters.
        """
        # Convert tools to OpenAI format
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        # Validate and process tool_choice
        if tool_choice is not None and tool_choice:
            if isinstance(tool_choice, str) and tool_choice not in ("auto", "any", "none"):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, dict) and len(formatted_tools) != 1:
                raise ValueError(
                    "If tool_choice is specified, exactly one tool must be provided."
                    f"Current number of tools: {len(formatted_tools)}"
                )
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError(
                        "tool_choice can be True only if there is one tool."
                        f"Current number of tools: {len(tools)}"
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