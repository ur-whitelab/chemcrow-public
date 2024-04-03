from typing import Any, Dict, List, Optional

from langchain.callbacks.streamlit.streamlit_callback_handler import (
    CHECKMARK_EMOJI,
    EXCEPTION_EMOJI,
    THINKING_EMOJI,
    LLMThought,
    LLMThoughtLabeler,
    LLMThoughtState,
    StreamlitCallbackHandler,
    ToolRecord,
)
from langchain_core.schema import AgentAction, AgentFinish, LLMResult
from streamlit.delta_generator import DeltaGenerator

from chemcrow.utils import is_smiles

from .utils import cdk


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
        if serialized["name"] == "Name2SMILES":
            safe_smiles = output.replace("[", "\[").replace("]", "\]")
            if is_smiles(output):
                self._container.markdown(
                    f"**{safe_smiles}**{cdk(output)}", unsafe_allow_html=True
                )

        if serialized["name"] == "ReactionPredict":
            rxn = f"{input_tool}>>{output}"
            safe_smiles = rxn.replace("[", "\[").replace("]", "\]")
            self._container.markdown(
                f"**{safe_smiles}**{cdk(rxn)}", unsafe_allow_html=True
            )

        if serialized["name"] == "ReactionRetrosynthesis":
            output = output.replace("[", "\[").replace("]", "\]")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        # Called with the name of the tool we're about to run (in `serialized[name]`),
        # and its input. We change our container's label to be the tool name.
        self._state = LLMThoughtState.RUNNING_TOOL
        tool_name = serialized["name"]
        self._last_tool = ToolRecord(name=tool_name, input_str=input_str)
        self._container.update(
            new_label=(
                self._labeler.get_tool_label(self._last_tool, is_complete=False)
                .replace("[", "\[")
                .replace("]", "\]")
            )
        )

        # Display note of potential long time
        if serialized["name"] == "ReactionRetrosynthesis" or serialized["name"] == "LiteratureSearch":
            self._container.markdown(
                f"‼️ Note: This tool can take some time to complete execution ‼️",
                unsafe_allow_html=True,
            )

    def complete(self, final_label: Optional[str] = None) -> None:
        """Finish the thought."""
        if final_label is None and self._state == LLMThoughtState.RUNNING_TOOL:
            assert (
                self._last_tool is not None
            ), "_last_tool should never be null when _state == RUNNING_TOOL"
            final_label = self._labeler.get_tool_label(
                self._last_tool, is_complete=True
            )
        self._state = LLMThoughtState.COMPLETE

        final_label = final_label.replace("[", "\[").replace("]", "\]")
        if self._collapse_on_complete:
            self._container.update(new_label=final_label, new_expanded=False)
        else:
            self._container.update(new_label=final_label)


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
            max_thought_containers=max_thought_containers,
            expand_new_thoughts=expand_new_thoughts,
            collapse_completed_thoughts=collapse_completed_thoughts,
            thought_labeler=thought_labeler,
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
            output_ph=self._output_placeholder,
            input_tool=self._last_input,
            serialized=self._serialized,
            **kwargs,
        )
        self._complete_current_thought()

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        if self._current_thought is not None:
            self._current_thought.complete(
                self._thought_labeler.get_final_agent_thought_label()
                .replace("[", "\[")
                .replace("]", "\]")
            )
            self._current_thought = None
