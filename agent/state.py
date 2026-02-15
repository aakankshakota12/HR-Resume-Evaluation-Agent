"""Agent state management."""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """All possible agent states."""
    INIT = "INIT"
    JOB_SETUP = "JOB_SETUP"
    EVALUATION = "EVALUATION"
    QUERY = "QUERY"


@dataclass
class StateContext:
    """
    Context that travels through state transitions.
    
    Represents the current state and context of the agent execution.
    """
    
    current_state: AgentState = AgentState.INIT
    job_description: Optional[str] = None
    query: Optional[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        """Validate state after creation."""
        if not isinstance(self.current_state, AgentState):
            raise ValueError(f"Invalid state: {self.current_state}")
    
    def transition_to(self, new_state: AgentState) -> None:
        """
        Transition to a new state.
        
        Args:
            new_state: Target AgentState
        
        Raises:
            ValueError: If state is invalid
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state transition to: {new_state}")
        
        logger.debug(f"State transition: {self.current_state.value} → {new_state.value}")
        self.current_state = new_state
    
    def set_error(self, error_msg: str) -> None:
        """
        Set error message.
        
        Args:
            error_msg: Error message string
        
        Raises:
            TypeError: If error_msg is not string
        """
        if not isinstance(error_msg, str):
            raise TypeError("Error message must be a string")
        
        self.error = error_msg
        logger.error(f"Agent error: {error_msg}")
    
    def clear_error(self) -> None:
        """Clear error."""
        self.error = None
    
    def is_error(self) -> bool:
        """Check if error state."""
        return self.error is not None