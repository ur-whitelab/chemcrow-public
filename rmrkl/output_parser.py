import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import langchain
from langchain import LLMChain
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException

from .prompts import (FINAL_ANSWER_ACTION, FORMAT_INSTRUCTIONS,
                      QUESTION_PROMPT, SUFFIX)


class ChatZeroShotOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if FINAL_ANSWER_ACTION in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )


        # Remove 'Thought' SUFFIX
        if text.startswith('Thought:'):
            text = text[8:]

        # \s matches against tab/newline/whitespace
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(action, action_input.strip(" ").strip('"'), text.strip())
