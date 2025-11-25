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
    def _translate_container_path_to_host(self, container_path: Path) -> Path:
        container_workspace = Path("/app")
        host_workspace = Path("/Users/arun/workspace/arun/ai-unifier-assesment")

        if not str(container_path).startswith(str(container_workspace)):
            return container_path

        relative_path = container_path.relative_to(container_workspace)
        host_path = host_workspace / relative_path
        return host_path

    def _execute_in_docker(self, working_dir: Path, timeout: int) -> CodeTesterOutput:
        DOCKER_IMAGE = "rust:1.70-slim"

        resolved_path = working_dir.resolve()
        host_path = self._translate_container_path_to_host(resolved_path)

        command = [
            "docker",
            "run",
            "--rm",
            "--volume",
            f"{host_path}:/app",
            "--workdir",
            "/app",
            "--network",
            "none",
            DOCKER_IMAGE,
            "cargo",
            "test",
            "--color",
            "never",
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

    def _ensure_rust_project_structure(self, working_dir: Path) -> None:
        src_dir = working_dir / "src"
        src_dir.mkdir(exist_ok=True)

        for rust_file in ["lib.rs", "main.rs"]:
            root_file = working_dir / rust_file
            if root_file.exists():
                src_file = src_dir / rust_file
                if not src_file.exists():
                    root_file.rename(src_file)

        cargo_toml = working_dir / "Cargo.toml"

        if cargo_toml.exists():
            return

        cargo_content = """[package]
name = "code_healing_test"
version = "0.1.0"
edition = "2021"

[dependencies]
"""
        cargo_toml.write_text(cargo_content)

    def test(self, input_data: CodeTesterInput) -> CodeTesterOutput:
        working_dir = Path(input_data.working_directory)

        if not working_dir.exists():
            error_msg = f"Working directory does not exist: {working_dir}"
            return CodeTesterOutput(success=False, stdout="", stderr=error_msg, exit_code=-1, command="")

        if input_data.language == "rust":
            self._ensure_rust_project_structure(working_dir)
            return self._execute_in_docker(working_dir, input_data.timeout)

        elif input_data.language == "python":
            command = _get_python_command()
            return self._execute_locally(command, working_dir, input_data.timeout)

        else:
            error_msg = f"Unsupported language: {input_data.language}"
            return CodeTesterOutput(success=False, stdout="", stderr=error_msg, exit_code=-1, command="")
