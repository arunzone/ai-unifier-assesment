import logging
import re
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from ai_unifier_assesment.agent.state import CodeHealingState
from ai_unifier_assesment.agent.tools.code_writer_tool import (
    CodeWriterInput,
    CodeWriterTool,
)
from ai_unifier_assesment.dependencies import get_settings

logger = logging.getLogger(__name__)


class CodeWriterService:
    def __init__(
        self,
        code_writer: Annotated[CodeWriterTool, Depends(CodeWriterTool)],
        settings: Annotated[object, Depends(get_settings)],
    ):
        self._code_writer = code_writer
        self._settings = settings

    def write_code_to_disk(self, state: CodeHealingState) -> CodeHealingState:
        logger.info("Writing code to disk...")

        if not state.current_code:
            logger.error("No code available to write")
            return state

        files = self._parse_code_files(state.current_code, state.language.value)

        if not files:
            logger.error("No files found in generated code")
            return state

        for filename, content in files.items():
            file_path = Path(state.working_directory) / filename
            logger.info(f"Writing {filename} ({len(content)} chars)")

            write_input = CodeWriterInput(
                code=content,
                file_path=str(file_path),
                language=state.language.value,
            )
            result = self._code_writer.write(write_input)

            if not result.success:
                logger.error(f"Failed to write {filename}: {result.message}")
            else:
                logger.info(f"✓ Wrote {filename}")

        return state

    def _parse_code_files(self, code_content: str, language: str) -> dict[str, str]:
        files = {}

        pattern = r"FILE:\s*(\S+)\s*```(?:\w+)?\s*\n(.*?)```"

        matches = re.finditer(pattern, code_content, re.DOTALL)

        for match in matches:
            filename = match.group(1).strip()
            content = match.group(2).strip()
            files[filename] = content
            logger.info(f"Parsed file: {filename} ({len(content)} chars)")

        if not files:
            logger.warning("No FILE: markers found, attempting to extract code blocks")
            files = self._fallback_parse(code_content, language)

        return files

    def _parse_python_code_blocks(self, code_blocks: list[str]) -> dict[str, str]:
        files = {}
        for block in code_blocks:
            if "import pytest" in block or "def test_" in block:
                files["test_main.py"] = block.strip()
            else:
                files["main.py"] = block.strip()
        return files

    def _parse_rust_code_blocks(self, code_blocks: list[str]) -> dict[str, str]:
        files = {}
        if code_blocks:
            files["lib.rs"] = code_blocks[0].strip()
        return files

    def _fallback_parse(self, code_content: str, language: str) -> dict[str, str]:
        code_blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)```", code_content, re.DOTALL)

        if language == "python":
            return self._parse_python_code_blocks(code_blocks)
        else:
            return self._parse_rust_code_blocks(code_blocks)
