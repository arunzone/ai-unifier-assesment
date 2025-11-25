import logging
import subprocess  # nosec B404: The tool's primary function requires executing controlled, external commands (pytest/cargo).
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


def _get_python_command() -> list[str]:
    return ["python", "-m", "pytest", "-v", "--tb=short", "--color=no"]


class CodeTesterTool:
    def _execute_in_docker(self, working_dir: Path, timeout: int) -> CodeTesterOutput:
        DOCKER_IMAGE = "rust:1.70-slim"
        TEST_COMMAND = "cargo test --color never"

        command = [
            "docker",
            "run",
            "--rm",
            "--volume",
            f"{working_dir.resolve()}:/app:ro",
            "--workdir",
            "/app",
            "--network",
            "none",
            DOCKER_IMAGE,
            TEST_COMMAND,
        ]

        command_str = " ".join(command)

        try:
            result = subprocess.run(  # nosec B603: shell=False explicitly set, command is controlled list
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )

            success = result.returncode == 0
            logger.info(f"Docker execution completed with exit code {result.returncode}, success={success}")

            return CodeTesterOutput(
                success=success,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                command=command_str,
            )

        except FileNotFoundError:
            error_msg = "Docker command not found. Ensure Docker is installed and running."
            logger.error(error_msg)
            return CodeTesterOutput(success=False, stdout="", stderr=error_msg, exit_code=-1, command=command_str)
        except subprocess.TimeoutExpired:
            error_msg = f"Docker test execution timed out after {timeout} seconds"
            logger.error(error_msg)
            return CodeTesterOutput(success=False, stdout="", stderr=error_msg, exit_code=-1, command=command_str)
        except Exception as e:
            error_msg = f"Unexpected error during Docker test execution: {e}"
            logger.error(error_msg)
            return CodeTesterOutput(success=False, stdout="", stderr=error_msg, exit_code=-1, command=command_str)

    def _execute_locally(self, command: list[str], working_dir: Path, timeout: int) -> CodeTesterOutput:
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
            logger.info(f"Local execution completed with exit code {result.returncode}, success={success}")

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

    def test(self, input_data: CodeTesterInput) -> CodeTesterOutput:
        working_dir = Path(input_data.working_directory)

        if not working_dir.exists():
            error_msg = f"Working directory does not exist: {working_dir}"
            return CodeTesterOutput(success=False, stdout="", stderr=error_msg, exit_code=-1, command="")

        if input_data.language == "rust":
            return self._execute_in_docker(working_dir, input_data.timeout)

        elif input_data.language == "python":
            command = _get_python_command()
            return self._execute_locally(command, working_dir, input_data.timeout)

        else:
            error_msg = f"Unsupported language: {input_data.language}"
            return CodeTesterOutput(success=False, stdout="", stderr=error_msg, exit_code=-1, command="")
