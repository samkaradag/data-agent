"""
This module provides utility functions for loading and managing configuration settings.

It handles reading configuration from a YAML file and making the settings available
throughout the application.  It also includes functions for setting the project ID
based on the environment.
"""
from agent.tools_context import vertexai_tools  # Import vertexai_tools here
import os
import yaml

_CONFIG = None

def get_vertexai_model():
    """
    Returns the Vertex AI model instance.
    """
    return vertexai_tools.model

def load_config(config_path="config.yaml"):
    """Loads configuration settings from a YAML file.

    Args:
        config_path: The path to the configuration YAML file.

    Returns:
        A dictionary containing the configuration settings.
        Returns an empty dictionary if the file is not found or an error occurs.
    """
    global _CONFIG
    if _CONFIG:  # Check if config is already loaded
        return _CONFIG

    try:
        with open(config_path, "r") as f:
            _CONFIG = yaml.safe_load(f)
            return _CONFIG
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return {}  # Return empty dictionary if file not found
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return {}  # Return empty dictionary if YAML error occurs
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}

def get_config():
    """Returns the loaded configuration settings.

    If the configuration has not been loaded yet, it will attempt to load it
    from the default path ("config.yaml").

    Returns:
        A dictionary containing the configuration settings, or an empty
        dictionary if loading fails.
    """
    global _CONFIG
    if not _CONFIG:
        load_config()  # Load if not already loaded
    return _CONFIG or {} # Return empty dict if loading still fails

def set_project_id_from_env(config):
    """Sets the project ID in the configuration based on the environment.

    Looks for environment variables like "PROJECT_ID" or "GOOGLE_CLOUD_PROJECT".
    If found, updates the "project_id" setting in the configuration.

    Args:
        config: The configuration dictionary.
    """
    project_id = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project_id:
        config["project_id"] = project_id