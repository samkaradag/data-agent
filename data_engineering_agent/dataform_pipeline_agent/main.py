"""
This module serves as the main entry point for running the interactive agent.

It imports and calls the `interactive_mode` function from the `agent_executor` module,
which handles the initialization and execution of the interactive agent loop,
allowing users to interact with the agent.
"""
from agent.agent_executor import interactive_mode

if __name__ == "__main__":
    interactive_mode()
