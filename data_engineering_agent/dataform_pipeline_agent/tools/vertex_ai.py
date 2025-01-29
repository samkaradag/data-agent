import json
import vertexai
from vertexai.generative_models import GenerativeModel
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda
from utils.tracers import trace_calls
from utils.prompt_loader import load_prompt # Import the function

class VertexAITools:
    @trace_calls
    def __init__(self, project_id, location="us-central1", model_name="gemini-2.0-flash-exp"):
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)

        self.model = GenerativeModel(model_name)
        self.project_id = project_id


    @trace_calls
    def structure_transformation_request(self, user_request):
        """
        Handles the user request using Vertex AI to extract details 
        and ask for missing information.

        Assumptions:
        - Relies on the LLM to accurately interpret the user request 
        and extract the relevant details.
        """
        from tools.bigquery import BigQueryTools
        bigquery_tools = BigQueryTools(project_id=self.project_id)
        # Load the prompt template
        prompt_template = load_prompt("structure_transformation_request")

        # Fill the prompt template
        prompt = prompt_template.format(user_request=user_request)

        # Call the Vertex AI model
        # print (f"Calling Vertex AI model {prompt}")
        response = self.model.generate_content(prompt)


        # Extract clean text content
        response_content = response.text.strip().replace("`json\n", "").replace("`", "")
        print(response_content)

        parsed_response = json.loads(
                response_content.replace("`json\n", "").replace("`", "").replace(" \n", "")
            )
        return parsed_response

    # Function to create a runnable tool from configuration
    def create_runnable_tool(self, tool_name, tool_config):
        tool_func = tool_config["function"]
        tool_input_model = tool_config["input_model"]
        tool_output_model = tool_config["output_model"]

        def _convert_to_langchain_tool_input(tool_input: dict) -> BaseModel:
            return tool_input_model(**tool_input)

        def _invoke_tool(tool_input: BaseModel) -> str:
            try:
                output = tool_func(**tool_input.dict())
                # Check if the output is a BaseModel and needs to be dumped into JSON
                if isinstance(output, BaseModel):
                    return output.model_dump_json()
                # Check if the output is a dictionary
                elif isinstance(output, dict):
                    # Directly return the output if it's a dictionary
                    return json.dumps(output)
                else:
                    # Handle other types of output
                    return json.dumps({"result": output})
            except Exception as e:
                print(f"Error invoking tool: {e}")
                return json.dumps({"error": str(e)})

        convert_to_input_model = RunnableLambda(_convert_to_langchain_tool_input)
        invoke_tool_lambda = RunnableLambda(_invoke_tool)

        runnable_tool = convert_to_input_model | invoke_tool_lambda | StrOutputParser()
        runnable_tool = runnable_tool.with_config(
            run_name=tool_name,
            name=tool_name,
        )

        return runnable_tool
