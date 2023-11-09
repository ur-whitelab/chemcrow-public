from langchain.callbacks.streamlit.streamlit_callback_handler import (
    LLMThoughtLabeler,
    StreamlitCallbackHandler,
    LLMThought,
)

from chemcrow.agents.prompts import FORMAT_INSTRUCTIONS, SUFFIX, QUESTION_PROMPT

from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Union

from langchain.callbacks.streamlit.mutable_expander import MutableExpander
from langchain.schema import AgentAction, AgentFinish, LLMResult

from streamlit.delta_generator import DeltaGenerator

import ast
from .utils import cdk, is_valid_smiles


class LLMThoughtChem(LLMThought):
    def __init__(
        self,
        parent_container: DeltaGenerator,
        labeler: LLMThoughtLabeler,
        expanded: bool,
        collapse_on_complete: bool,
    ):
        super().__init__(
            parent_container,
            labeler,
            expanded,
            collapse_on_complete,
        )

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        output_ph: dict = {},
        input_tool: str = "",
        serialized: dict = {},
        **kwargs: Any,
    ) -> None:

        # Depending on the tool name, decide what to display.

        if serialized['name'] == 'Name2SMILES':
            self._container.markdown(
                f"**{output}**{cdk(output)}",
                unsafe_allow_html=True
            )

        if serialized['name'] == 'RXNPredict':
            rxn = f"{input_tool}>>{output}"
            self._container.markdown(
                f"**{output}**{cdk(rxn)}",
                unsafe_allow_html=True
            )

class StreamlitCallbackHandlerChem(StreamlitCallbackHandler):
    def __init__(
        self,
        parent_container: DeltaGenerator,
        *,
        max_thought_containers: int = 4,
        expand_new_thoughts: bool = True,
        collapse_completed_thoughts: bool = True,
        thought_labeler: Optional[LLMThoughtLabeler] = None,
        output_placeholder: dict = {},
    ):
        super(StreamlitCallbackHandlerChem, self).__init__(
            parent_container,
            max_thought_containers = max_thought_containers,
            expand_new_thoughts = expand_new_thoughts,
            collapse_completed_thoughts = collapse_completed_thoughts,
            thought_labeler = thought_labeler
        )

        self._output_placeholder = output_placeholder
        self.last_input = ""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        if self._current_thought is None:
            self._current_thought = LLMThoughtChem(
                parent_container=self._parent_container,
                expanded=self._expand_new_thoughts,
                collapse_on_complete=self._collapse_completed_thoughts,
                labeler=self._thought_labeler,
            )

        self._current_thought.on_llm_start(serialized, prompts)

        # We don't prune_old_thought_containers here, because our container won't
        # be visible until it has a child.

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        self._require_current_thought().on_tool_start(serialized, input_str, **kwargs)
        self._prune_old_thought_containers()
        self._last_input = input_str
        self._serialized = serialized

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:

        self._require_current_thought().on_tool_end(
            output,
            color,
            observation_prefix,
            llm_prefix,
            output_ph = self._output_placeholder,
            input_tool = self._last_input,
            serialized = self._serialized,
            **kwargs
        )
        self._complete_current_thought()
