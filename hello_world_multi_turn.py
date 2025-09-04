"""
Simple Multi-Turn Hello World Environment

Does the following,
1. How to inherit from MultiTurnEnv
2. How to implement the required abstract methods
3. Basic conversation flow with completion conditions
"""

from datasets import Dataset
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Messages, State
import verifiers as vf
import logging


class HelloWorldMultiTurnEnv(MultiTurnEnv):
    """
    The class for the Environment that details the interaction protocol.

    The environment hardcodes a few conversation steps as an example in `self.conversation_steps`.

    The overridden `self.is_completed` method determines when the multi-turn chat conversation is complete.
    """

    def __init__(self, max_turns: int = 5, **kwargs):
        super().__init__(max_turns=max_turns, **kwargs)

        self.conversation_steps = [
            "Hello! What's your name?",
            "Nice to meet you! How has your day been?",
            "Nice weather we are having!",
            "I like coffee!",
            "That's great to hear! Thanks for chatting with me. Goodbye!"
        ]

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """
        Determine if the conversation has ended.

        This env has a few ways to end the ongoing conversation.
        """
        # Check if we've reached max turns.
        turn = state.get("turn", 0)

        if turn == 0:
            return False

        if turn >= self.max_turns:  # In this example `max_turns` should match `len(self.conversation_steps)`.
            return True

        # Check if user said goodbye using simple keyword detection.
        # Note that we have to check if the messages object is in the OAI chat format.
        if isinstance(messages, list):
            if messages and messages[-1].get("role") == "assistant":
                last_message = messages[-1].get("content", "").lower()
            elif len(messages) >= 2:
                last_message = messages[-2].get("content", "").lower()
            else:
                last_message = f"Unable to parse {messages}. Goodbye."  # Note, this is an error state
        else:
            last_message = str(messages).lower()

        if "goodbye" in last_message or "bye" in last_message:
            return True

        if "Unable to parse" in last_message:
            return True

        # Complete after we've gone through all conversation steps
        return False

    async def env_response(
            self, messages: Messages, state: State, **kwargs
    ) -> tuple[Messages, State]:
        """
        Env responds to the current conversation thread.
        """
        logger = logging.getLogger(__name__)

        logger.info("messages")

        turn = state.get("turn", 0)

        # Get the appropriate response based on the current turn
        if turn < len(self.conversation_steps):
            response_text = self.conversation_steps[turn]
        else:
            # Fallback response if we somehow exceed our planned steps
            response_text = "Thanks for chatting! Goodbye!"

        # Format response based on message type
        if isinstance(messages, list):
            # Chat message format
            env_messages = [{"role": "user", "content": response_text}]
        else:
            # String format
            env_messages = f"\n\nEnvironment: {response_text}\n\nYour response: "

        state["last_env_response"] = response_text

        return env_messages, state


def load_environment(
        dataset_name: str = "won't be used in this example",
        dataset_split: str = "won't be used in this example",
        system_prompt: str | None = "Greet the user and have a conversation with them! Respond to them with <bot_resp> tags.",
) -> vf.Environment:

    parser = vf.XMLParser(["bot_resp"], answer_field="bot_resp")

    def chat_reward(prompt, completion, answer, state, info, task, parser):
        bot_resp = parser.parse(completion)

        if "Goodbye" in bot_resp or "Unable to parse" in bot_resp:
            return 0.0
        return 1.0

    rubric = vf.Rubric(
        funcs=[chat_reward],
        weights=[1.0],
    )

    dummy_data = {
        'prompt': [[{"role": "system", "content": "You are a friendly assistant."}]],
        'answer': ["Hello!"]
    }

    vf_env = HelloWorldMultiTurnEnv(
        dataset=Dataset.from_dict(dummy_data),
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_turns=5,
    )

    return vf_env