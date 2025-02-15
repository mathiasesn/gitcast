import os
from textwrap import dedent

from openai import OpenAI


class Podcast:
    def __init__(
        self, model: str = "gpt-4o", max_content_length: int = 128_000
    ) -> None:
        self.model = model
        self.max_content_length = max_content_length
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @property
    def system_prompt(self) -> str:
        return dedent("""
            You are a creative podcast producer.
            Generate a transcript for a podcast episode where two hosts, Alex and Jordan,
            have an engaging, natural-sounding conversation. Their discussion is based on a GitHub repository,
            where they:
              1. Introduce the project using its README.
              2. Give an overview of the repository structure.
              3. Dive deep into the code by analyzing key parts.
            Make sure the transcript has a clear flow with natural back-and-forth dialogue.
        """)

    def _generate_messages(self, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "developer", "content": self.system_prompt},
            {"role": "user", "content": dedent(user_prompt)},
        ]

    def _call_llm(self, user_prompt: str) -> str:
        messages = self._generate_messages(user_prompt)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content

    def _chunk_snippets(self, content: list[str]) -> list[str]:
        chunks = []
        current_chunk = ""
        for idx, snippet in enumerate(content):
            snippet_text = f"\n--- Snippet {idx + 1} ---\n{snippet}\n"
            # If adding the next snippet exceeds allowed context, start a new chunk.
            if len(current_chunk) + len(snippet_text) > self.max_content_length:
                chunks.append(current_chunk)
                current_chunk = snippet_text
            else:
                current_chunk += snippet_text

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def generate_transcript(
        self, readme: str, tree_structure: str, content: list[str]
    ) -> str:
        transcript_parts = []

        # Generate introduction from the README.
        prompt_intro = f"""
            Based on the following README content, generate an engaging introduction for a podcast episode.
            The introduction should introduce the repository, its purpose, and set the stage for a deep technical discussion.
            README:
            {readme}
        """
        introduction = self._call_llm(prompt_intro)
        transcript_parts.append("=== Introduction ===\n" + introduction)

        # Generate an overview from the repository tree structure.
        prompt_overview = f"""
            Given the repository tree structure below, create an overview segment for a podcast episode.
            Discuss the organization, main features, and structure of the project.
            Repository Tree Structure:
            {tree_structure}
        """
        overview = self._call_llm(prompt_overview)
        transcript_parts.append("=== Repository Overview ===\n" + overview)

        # Generate a deep dive into the code with context splitting if necessary.
        deep_dive_parts = []
        base_prompt = """
            Now, analyze the following key code snippets from the repository.
            Produce a deep dive discussion for a podcast episode that examines the design, functionality,
            and any interesting details of the implementation.
            Code Snippets:
        """

        # Chunk the code snippets so that each group fits within the allowed context length.
        chunks = self._chunk_snippets(content)
        for chunk in chunks:
            prompt_deep_dive = base_prompt + chunk
            response = self._call_llm(prompt_deep_dive)
            deep_dive_parts.append(response)

        # If multiple deep dive parts were generated, have the model synthesize them into one coherent discussion.
        if len(deep_dive_parts) > 1:
            synthesis_prompt = "Combine the following deep dive discussions into one coherent deep dive discussion for a podcast episode:\n"
            for idx, part in enumerate(deep_dive_parts):
                synthesis_prompt += f"\n--- Part {idx + 1} ---\n{part}\n"
            deep_dive = self._call_llm(synthesis_prompt)
        else:
            deep_dive = deep_dive_parts[0] if deep_dive_parts else ""

        transcript_parts.append("=== Deep Dive into the Code ===\n" + deep_dive)

        # Combine all parts into the final transcript.
        transcript = "\n\n".join(transcript_parts)
        return transcript
