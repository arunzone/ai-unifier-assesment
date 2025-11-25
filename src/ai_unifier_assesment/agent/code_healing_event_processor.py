"""Event stream processor for code healing agent.

Processes LangGraph node events and converts them into structured SSE events.
Uses Strategy Pattern to avoid high cyclomatic complexity.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict


logger = logging.getLogger(__name__)


class NodeEventMapper(ABC):
    @abstractmethod
    def map(self, updates: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Map node updates to event type and data.

        Args:
            updates: Dict of fields updated by the node

        Returns:
            Tuple of (event_type, event_data)
        """


class LanguageDetectedMapper(NodeEventMapper):
    def map(self, updates: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        return "language_detected", {"language": updates.get("language", "")}


class WorkdirSetupMapper(NodeEventMapper):
    def map(self, updates: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        return "workdir_setup", {"working_directory": updates.get("working_directory", "")}


class CodeGeneratedMapper(NodeEventMapper):
    def map(self, updates: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        return "code_generated", {"code_length": len(updates.get("current_code", ""))}


class CodeWrittenMapper(NodeEventMapper):
    def map(self, updates: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        return "code_written", {}


class TestResultMapper(NodeEventMapper):
    def map(self, updates: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        if updates.get("success", False):
            return "tests_passed", {"message": updates.get("final_message", "")}

        test_output = updates.get("test_output", "")
        preview = test_output[:500] + "..." if len(test_output) > 500 else test_output
        return "tests_failed", {"error_preview": preview}


class RetryMapper(NodeEventMapper):
    def map(self, updates: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        return "retry", {"next_attempt": updates.get("attempt_number", 0) + 1}


class FinalizeMapper(NodeEventMapper):
    def map(self, updates: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        final_message = updates.get("final_message", "")
        event_type = "success" if final_message.startswith("Success!") else "failure"
        return event_type, {"message": final_message}


class CodeHealingEventProcessor:
    def __init__(self, max_attempts: int = 3):
        self._mappers: Dict[str, NodeEventMapper] = {
            "detect_language": LanguageDetectedMapper(),
            "setup_workdir": WorkdirSetupMapper(),
            "code_generator": CodeGeneratedMapper(),
            "write_code": CodeWrittenMapper(),
            "run_tests": TestResultMapper(),
            "increment_attempt": RetryMapper(),
            "finalize": FinalizeMapper(),
        }

    async def process_graph_event(self, event: tuple):
        updates = event[1] if len(event) > 1 else {}

        for node_name, update_dict in updates.items():
            mapper = self._mappers.get(node_name)
            if not mapper:
                continue

            event_type, event_data = mapper.map(update_dict)
            if event_type:
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(event_data)}\n\n"
