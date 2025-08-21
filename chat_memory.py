from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SlidingWindowMemory:
    """A tiny sliding-window memory that stores the last N turns.
    
    A 'turn' is a User + Bot pair, but we store messages as a flat list of (role, text).
    """
    max_turns: int = 4
    messages: List[Tuple[str, str]] = field(default_factory=list)

    def add(self, role: str, text: str) -> None:
        self.messages.append((role, text))
        # Keep only the last max_turns*2 messages (User+Bot per turn)
        max_msgs = self.max_turns * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

    def add_user(self, text: str) -> None:
        self.add("User", text)

    def add_bot(self, text: str) -> None:
        self.add("Bot", text)

    def build_prompt(self, new_user_message: str) -> str:
        """Build a chat-style prompt from recent memory + the new user message."""
        history_lines = []
        for role, text in self.messages:
            history_lines.append(f"{role}: {text}")
        history = "\n".join(history_lines)
        prompt = (
            "You are a helpful assistant. Answer briefly and clearly.\n"
            "Use the recent conversation to stay on topic.\n\n"
        )
        if history:
            prompt += f"Conversation so far:\n{history}\n\n"
        prompt += f"User: {new_user_message}\nAssistant:"
        return prompt

    def clear(self) -> None:
        self.messages.clear()