import logging
import shutil
import subprocess  # nosec B404: The tool's primary function requires executing controlled, external commands (pytest/cargo).
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CodeTesterInput(BaseModel):
    working_directory: str = Field(description="Directory where tests should be executed")
    language: Literal["python", "rust"] = Field(description="Programming language (python or rust)")
    timeout: int = Field(default=30, description="Timeout in seconds for test execution")


class CodeTesterOutput(BaseModel):
    success: bool = Field(description="Whether all tests passed")
    stdout: str = Field(description="Standard output from test execution")
    stderr: str = Field(description="Standard error from test execution")
    exit_code: int = Field(description="Exit code from test command")
    command: str = Field(description="Command that was executed")


class CodeTesterTool:
    @staticmethod
    def _try_pytest_module() -> list[str] | None:
        """Try to find pytest as a Python module."""
        try:
            result = subprocess.run(  # nosec B603: Command components are hardcoded or from sys.executable.
                [sys.executable, "-m", "pytest", "--version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.info(f"Found pytest via Python module: {sys.executable} -m pytest")
                return [sys.executable, "-m", "pytest"]
        except Exception as e:
            logger.debug(f"pytest module not found in current Python: {e}")
        return None

    @staticmethod
    def _try_pytest_path() -> list[str] | None:
        """Try to find pytest in PATH."""
        pytest_path = shutil.which("pytest")
        if pytest_path:
            logger.info(f"Found pytest in PATH: {pytest_path}")
            return [pytest_path]
        return None

    @staticmethod
    def _try_venv_paths() -> list[str] | None:
        """Try to find pytest in common virtual environment locations."""
        venv_paths = [
            ".venv/bin/python",
            "venv/bin/python",
            "env/bin/python",
        ]

        for venv_python in venv_paths:
            if Path(venv_python).exists():
                try:
                    result = subprocess.run(  # nosec B603: Command components are hardcoded or validated path.
                        [venv_python, "-m", "pytest", "--version"],
                        capture_output=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        logger.info(f"Found pytest in virtual environment: {venv_python}")
                        return [venv_python, "-m", "pytest"]
                except (FileNotFoundError, OSError, subprocess.TimeoutExpired, ValueError) as e:
                    logger.debug(f"Failed to run pytest in {venv_python}: {e}")
                    continue
        return None

    @staticmethod
    def _find_pytest() -> list[str]:
        """Find pytest command using multiple strategies."""
        # Strategy 1: Use current Python interpreter with pytest module
        pytest_cmd = CodeTesterTool._try_pytest_module()
        if pytest_cmd:
            return pytest_cmd

        # Strategy 2: Look for pytest in PATH
        pytest_cmd = CodeTesterTool._try_pytest_path()
        if pytest_cmd:
            return pytest_cmd

        # Strategy 3: Check common virtual environment locations
        pytest_cmd = CodeTesterTool._try_venv_paths()
        if pytest_cmd:
            return pytest_cmd

        # If nothing works, return the module approach and let it fail with a clear message
        logger.warning("pytest not found in any standard location")
        return [sys.executable, "-m", "pytest"]

    @staticmethod
    def _create_error_output(error_msg: str, command: str) -> CodeTesterOutput:
        """Create a standardized error output."""
        logger.error(error_msg)
        return CodeTesterOutput(
            success=False,
            stdout="",
            stderr=error_msg,
            exit_code=-1,
            command=command,
        )

    @staticmethod
    def _execute_command(command: list[str], working_dir: Path, timeout: int) -> CodeTesterOutput:
        """Execute test command and handle exceptions."""
        try:
            logger.info(f"Running command: {' '.join(command)} in {working_dir}")

            result = subprocess.run(  # nosec B603: Command is fully controlled by the tool logic.
                command,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            success = result.returncode == 0
            logger.info(f"Test execution completed with exit code {result.returncode}, success={success}")

            return CodeTesterOutput(
                success=success,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                command=" ".join(command),
            )

        except subprocess.TimeoutExpired:
            error_msg = f"Test execution timed out after {timeout} seconds"
            return CodeTesterTool._create_error_output(error_msg, " ".join(command))
        except FileNotFoundError as e:
            error_msg = (
                f"Test command not found: {e}. Ensure {'cargo' if 'cargo' in command else 'pytest'} is installed."
            )
            return CodeTesterTool._create_error_output(error_msg, " ".join(command))
        except Exception as e:
            error_msg = f"Unexpected error during test execution: {e}"
            return CodeTesterTool._create_error_output(error_msg, " ".join(command))

    @staticmethod
    def _build_command(language: str) -> list[str]:
        """Build test command based on language."""
        if language == "rust":
            return ["cargo", "test", "--color", "never"]
        else:  # python
            pytest_cmd = CodeTesterTool._find_pytest()
            return pytest_cmd + ["-v", "--tb=short", "--color=no"]

    def test(self, input_data: CodeTesterInput) -> CodeTesterOutput:
        working_dir = Path(input_data.working_directory)

        # Validate working directory exists
        if not working_dir.exists():
            error_msg = f"Working directory does not exist: {working_dir}"
            return CodeTesterOutput(
                success=False,
                stdout="",
                stderr=error_msg,
                exit_code=-1,
                command="",
            )

        command = self._build_command(input_data.language)
        return self._execute_command(command, working_dir, input_data.timeout)
