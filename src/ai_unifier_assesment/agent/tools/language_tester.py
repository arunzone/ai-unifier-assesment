from pathlib import Path
from typing import Protocol

from ai_unifier_assesment.agent.tools.tester_models import CodeTesterOutput


class LanguageTester(Protocol):
    def prepare_working_directory(self, working_dir: Path) -> None: ...

    def run_tests(self, working_dir: Path, timeout: int) -> CodeTesterOutput: ...
