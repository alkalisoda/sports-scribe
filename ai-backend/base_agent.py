"""Base agent class for the multi-agent system."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the base agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        self.config = config or {}
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the agent with configuration.
        
        Args:
            config: Configuration dictionary
        """
        pass
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Any:
        """Execute a task using the agent.
        
        Args:
            task: Task dictionary containing parameters
            
        Returns:
            Task result
        """
        pass
    
    @abstractmethod
    def finalize(self) -> None:
        """Clean up resources when agent is done."""
        pass
    
    def get_name(self) -> str:
        """Get the agent name.
        
        Returns:
            Agent name
        """
        return self.__class__.__name__
    
    def get_config(self) -> Dict[str, Any]:
        """Get the agent configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()