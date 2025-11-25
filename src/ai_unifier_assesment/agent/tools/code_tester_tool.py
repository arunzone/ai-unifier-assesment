import logging
from pathlib import Path

from ai_unifier_assesment.agent.tools.language_tester import LanguageTester
from ai_unifier_assesment.agent.tools.python_tester import PythonTester
from ai_unifier_assesment.agent.tools.rust_tester import RustTester
from ai_unifier_assesment.agent.tools.tester_models import (
    CodeTesterInput,
    CodeTesterOutput,
)

logger = logging.getLogger(__name__)


class CodeTesterTool:
    def __init__(self) -> None:
        self._testers: dict[str, LanguageTester] = {
            "python": PythonTester(),
            "rust": RustTester(),
        }

    def test(self, input_data: CodeTesterInput) -> CodeTesterOutput:
        working_dir = Path(input_data.working_directory)

        if not working_dir.exists():
            error_msg = f"Working directory does not exist: {working_dir}"
            return CodeTesterOutput(success=False, stdout="", stderr=error_msg, exit_code=-1, command="")

        tester = self._testers.get(input_data.language)
        if not tester:
            error_msg = f"Unsupported language: {input_data.language}"
            return CodeTesterOutput(success=False, stdout="", stderr=error_msg, exit_code=-1, command="")

        tester.prepare_working_directory(working_dir)
        result: CodeTesterOutput = tester.run_tests(working_dir, input_data.timeout)
        return result
