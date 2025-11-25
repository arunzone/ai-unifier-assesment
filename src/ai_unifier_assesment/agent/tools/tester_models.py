from typing import Literal

from pydantic import BaseModel, Field


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
