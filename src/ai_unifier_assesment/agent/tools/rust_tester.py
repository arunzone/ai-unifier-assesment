import logging
import subprocess  # nosec B404: subprocess required for running cargo tests in Docker container
from pathlib import Path

from ai_unifier_assesment.agent.tools.tester_models import CodeTesterOutput

logger = logging.getLogger(__name__)


class RustTester:
    DOCKER_IMAGE = "rust:1.70-slim"
    CONTAINER_WORKSPACE = Path("/app")
    HOST_WORKSPACE = Path("/Users/arun/workspace/arun/ai-unifier-assesment")

    def prepare_working_directory(self, working_dir: Path) -> None:
        self._ensure_rust_project_structure(working_dir)

    def run_tests(self, working_dir: Path, timeout: int) -> CodeTesterOutput:
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
            self.DOCKER_IMAGE,
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

    def _translate_container_path_to_host(self, container_path: Path) -> Path:
        if not str(container_path).startswith(str(self.CONTAINER_WORKSPACE)):
            return container_path

        relative_path = container_path.relative_to(self.CONTAINER_WORKSPACE)
        host_path = self.HOST_WORKSPACE / relative_path
        return host_path

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
