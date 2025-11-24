"""Code Writer Tool - Writes generated code to disk."""

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CodeWriterInput(BaseModel):
    """Input schema for code writing."""

    code: str = Field(description="The complete code to write to file")
    file_path: str = Field(description="Path where the code file should be written")
    language: Literal["python", "rust"] = Field(description="Programming language (python or rust)")


class CodeWriterOutput(BaseModel):
    """Output schema for code writing."""

    success: bool = Field(description="Whether the file was written successfully")
    file_path: str = Field(description="Path where the file was written")
    message: str = Field(description="Success or error message")


class CodeWriterTool:
    """Tool for writing generated code to disk.

    This tool writes code files to the specified path, creating parent
    directories as needed. It supports Python (.py) and Rust (.rs) files.
    """

    @staticmethod
    def _validate_file_extension(file_path: Path, language: str) -> CodeWriterOutput | None:
        """Validate that file extension matches the language."""
        expected_ext = ".py" if language == "python" else ".rs"
        if file_path.suffix != expected_ext:
            return CodeWriterOutput(
                success=False,
                file_path=str(file_path),
                message=f"File extension {file_path.suffix} does not match language {language}",
            )
        return None

    @staticmethod
    def _write_code_file(file_path: Path, code: str) -> CodeWriterOutput:
        """Write code to file and handle exceptions."""
        try:
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the code to file
            file_path.write_text(code, encoding="utf-8")

            logger.info(f"Successfully wrote code to {file_path}")
            return CodeWriterOutput(
                success=True,
                file_path=str(file_path),
                message=f"Code written successfully to {file_path}",
            )

        except PermissionError as e:
            error_msg = f"Permission denied writing to {file_path}: {e}"
            logger.error(error_msg)
            return CodeWriterOutput(
                success=False,
                file_path=str(file_path),
                message=error_msg,
            )
        except OSError as e:
            error_msg = f"OS error writing to {file_path}: {e}"
            logger.error(error_msg)
            return CodeWriterOutput(
                success=False,
                file_path=str(file_path),
                message=error_msg,
            )
        except Exception as e:
            error_msg = f"Unexpected error writing code: {e}"
            logger.error(error_msg)
            return CodeWriterOutput(
                success=False,
                file_path=str(file_path),
                message=error_msg,
            )

    def write(self, input_data: CodeWriterInput) -> CodeWriterOutput:
        """Write code to disk.

        Args:
            input_data: Contains code content, file path, and language

        Returns:
            CodeWriterOutput with success status and message
        """
        file_path = Path(input_data.file_path)

        # Validate file extension matches language
        validation_error = self._validate_file_extension(file_path, input_data.language)
        if validation_error:
            return validation_error

        # Write the code to file
        return self._write_code_file(file_path, input_data.code)
