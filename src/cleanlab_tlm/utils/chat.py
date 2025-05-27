"""Utilities for formatting chat messages into prompt strings.

This module provides helper functions for working with chat messages in the format used by
OpenAI's chat models.
"""


def form_prompt_string(messages: list[dict[str, str]]) -> str:
    """
    Convert a list of chat messages into a single string prompt.

    If there is only one message, returns the content directly. Otherwise, concatenates
    all messages with appropriate role prefixes and ends with "Assistant:" to indicate
    the assistant's turn is next.

    Args:
        messages (List[Dict[str, str]]): A list of dictionaries representing chat messages.
            Each dictionary should contain 'role' and 'content', just as in OpenAI's
            `messages` format.

    Returns:
        str: A formatted string representing the chat history as a single prompt.

    Example:
        ```python
        messages = [
            {"role": "user", "content": "Can you explain how photosynthesis works?"},
            {
                "role": "assistant",
                "content": "Of course! Photosynthesis is the process...",
            },
            {"role": "user", "content": "Can you summarize that in one sentence?"},
        ]
        result = form_prompt_string(messages)
        print(result)
        # User: Can you explain how photosynthesis works?
        #
        # Assistant: Of course! Photosynthesis is the process...
        #
        # User: Can you summarize that in one sentence?
        #
        # Assistant:
        ```
    """
    if len(messages) == 1:
        return messages[0]["content"]

    output = ""
    for msg in messages:
        role = msg.get("name", msg["role"])
        if role == "system":
            prefix = ""
        elif role == "user":
            prefix = "User: "
        elif role == "assistant":
            prefix = "Assistant: "
        else:
            prefix = role.capitalize() + ": "
        output += f"{prefix}{msg['content']}\n\n"

    output += "Assistant:"
    return output.strip()
