import logging
import subprocess  # nosec B404: subprocess required for running pytest tests in controlled environment
from pathlib import Path

from ai_unifier_assesment.agent.tools.tester_models import CodeTesterOutput

logger = logging.getLogger(__name__)


class PythonTester:
    def prepare_working_directory(self, working_dir: Path) -> None:
        pass

    def run_tests(self, working_dir: Path, timeout: int) -> CodeTesterOutput:
        command = ["python", "-m", "pytest", "-v", "--tb=short", "--color=no"]
        command_str = " ".join(command)

        try:
            result = subprocess.run(  # nosec B603: shell=False explicitly set, command is controlled list
                command,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )

            success = result.returncode == 0

            return CodeTesterOutput(
                success=success,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                command=command_str,
            )

        except Exception as e:
            error_msg = f"Local execution error: {e}"
            logger.error(error_msg)
            return CodeTesterOutput(success=False, stdout="", stderr=error_msg, exit_code=-1, command=command_str)
