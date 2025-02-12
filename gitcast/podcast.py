import os
from textwrap import dedent

from openai import OpenAI


class Podcast:
    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @property
    def system_prompt(self) -> str:
        return """
You are a creative podcast producer.
Generate a transcript for a podcast episode where two hosts, Alex and Jordan,
have an engaging, natural-sounding conversation discussing the future of artificial intelligence
and its impact on society. The conversation should include introductions, back-and-forth dialogue,
and reflections on current trends and potential future developments.
        """

    def _generate_messages(self, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "developer", "content": dedent(self.system_prompt)},
            {"role": "user", "content": dedent(user_prompt)},
        ]

    def _call_llm(self, user_prompt: str) -> str:
        messages = self._generate_messages(user_prompt)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content

    def generate_transcript(
        self, readme: str, tree_structure: str, content: list[str]
    ) -> str:
        # Create introduction from readme
        prompt = f"Create an introduction for a podcast episode based on the following README:\n{readme}\n"
        introduction = self._call_llm(prompt)  # noqa: F841
