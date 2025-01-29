"""
This module defines Pydantic models for input and output schemas used by LangChain tools.

These schemas are used to structure the data passed to and returned from the various
tools used by the agent, ensuring type safety and facilitating validation.  They
define the expected format and content for each tool's input and output.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Define tool schemas for LangChain

class InformationSchemaInput(BaseModel):
    """Input schema for querying the information schema."""
    dataset_name: Optional[str] = Field(None, description="The name of the dataset to query")
    table_name: Optional[str] = Field(None, description="The name of the table to query")


class InformationSchemaOutput(BaseModel):
    """Output schema for the information schema query results."""
    result: List[Dict[str, Any]] = Field(...,
        description="The result of the information schema query")


class FindRelevantDatasetInput(BaseModel):
    """Input schema for finding a relevant dataset."""
    table_name: str = Field(
        ..., description="The name of the table to find the relevant dataset for"
    )
    user_request: str = Field(..., description="The original user request")


class FindRelevantDatasetOutput(BaseModel):
    """Output schema for the find relevant dataset operation."""
    dataset: Optional[str] = Field(None, description="The name of the relevant dataset")


class ParseLLMOutputInput(BaseModel):
    """Input schema for parsing LLM output."""
    llm_output: str = Field(..., description="The output from the LLM to be parsed")


class ParseLLMOutputOutput(BaseModel):
    """Output schema for the parsed LLM output."""
    files: List[Dict[str, Any]] = Field(
        ..., description="A JSON object containing file paths and contents"
    )


class UploadAndCompileFilesInput(BaseModel):
    """Input schema for uploading and compiling files."""
    files: List[Dict[str, Any]] = Field(..., description="A list of files to upload and compile")
    workspace_name: str = Field(..., description="The name of the Dataform workspace")


class UploadAndCompileFilesOutput(BaseModel):
    """Output schema for the upload and compile files operation."""
    compilation_results: Dict = Field(..., description="The Dataform compilation results")


class FixCompilationErrorsInput(BaseModel):
    """Input schema for fixing compilation errors."""
    files: List[Dict[str, Any]] = Field(..., description="The current files")
    errors: Dict = Field(..., description="The compilation errors")


class FixCompilationErrorsOutput(BaseModel):
    """Output schema for the fix compilation errors operation."""
    files: List[Dict[str, Any]] = Field(..., description="The fixed files")


class HandleUserRequestInput(BaseModel):
    """Input schema for handling a user request."""
    user_request: str = Field(..., description="The user's request")


class HandleUserRequestOutput(BaseModel):
    """Output schema for the handle user request operation."""
    parsed_request: Optional[Dict[str, Any]] = Field(None, description="Parsed request")
    follow_up: Optional[str] = Field(None, description="Follow-up question")
    error: Optional[str] = Field(None, description="Error message")
    target_tables: Optional[List[Dict[str, str]]] = Field(None, description="Target tables")


class GeneratePipelineCodeInput(BaseModel):
    """Input schema for generating pipeline code."""
    source_tables: List[Dict[str, Any]] = Field(..., description="The source tables")
    target_tables: Optional[List[Dict]] = Field(
        None, description="The target tables for dimensions and facts"
    )  # Now optional
    transformations: Dict[str, Any] = Field(..., description="The transformations to apply")
    intermediate_tables: Optional[List[Dict]] = Field(None, description="The intermediate tables")
    data_quality_checks: Optional[Dict] = Field(
        None, description="Data quality checks to be applied"
    )


class GeneratePipelineCodeOutput(BaseModel):
    """Output schema for the generate pipeline code operation."""
    pipeline_code: str = Field(..., description="The generated pipeline code")


class RefineLLMResponseInput(BaseModel):
    """Input schema for refining the LLM response."""
    prompt: str = Field(..., description="The original prompt")
    previous_response: str = Field(..., description="The previous LLM response")
    error: str = Field(..., description="The error encountered")


class RefineLLMResponseOutput(BaseModel):
    """Output schema for the refine LLM response operation."""
    refined_response: str = Field(..., description="The refined LLM response")


class AskForClarificationsInput(BaseModel):
    """Input schema for asking for clarifications."""
    user_request: str = Field(..., description="The user's request")
    missing_information: Optional[List[str]] = Field(
        None,
        description="""List of missing information identified in the user request
        or in the previous turns""",
    )


class AskForClarificationsOutput(BaseModel):
    """Output schema for the ask for clarifications operation."""
    follow_up_questions: List[str] = Field(..., description="Follow-up clarification questions")


class ValidateDataInput(BaseModel):
    """Input schema for validating data."""
    table: str = Field(..., description="The table to validate")
    validation_rules: Dict = Field(..., description="The validation rules")


class ValidateDataOutput(BaseModel):
    """Output schema for the validate data operation."""
    validation_results: List[Dict] = Field(..., description="The validation results")
