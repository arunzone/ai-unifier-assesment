from functools import lru_cache
from pathlib import Path


class PromptLoader:
    def __init__(self) -> None:
        self._prompts_dir = Path(__file__).parent

    @lru_cache(maxsize=32)
    def load(self, prompt_name: str) -> str:
        prompt_path = self._prompts_dir / f"{prompt_name}.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text().strip()


prompt_loader = PromptLoader()
