# flake8: noqa

PREFIX = """
You are an AI system.
"""
FORMAT_INSTRUCTIONS = """
You should only respond with a single complete
Thought, Action, Action Input format
OR a single Final Answer format.

Complete Format:

Thought: (reflect on your progress and decide what to do next)
Action: (the action name, should be one of [{tool_names}])
Action Input: (the input string to the action)

OR

Final Answer: (the final answer to the original input question)

"""
QUESTION_PROMPT = """
Answer the question below using the following tools:

{tool_strings}

Question: {input}
"""
SUFFIX = """
Thought: {agent_scratchpad}
"""
FINAL_ANSWER_ACTION = "Final Answer:"
