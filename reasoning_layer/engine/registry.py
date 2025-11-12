"""Tool registry for managing and executing tools."""
from typing import Callable, Dict


class Tool:
    """A tool that can be executed with arguments."""
    
    def __init__(self, name: str, description: str, fn: Callable[[dict], dict]):
        self.name = name
        self.description = description
        self.fn = fn
    
    def __call__(self, args: dict) -> dict:
        """Execute the tool with given arguments."""
        return self.fn(args)


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Tool:
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]
    
    def specs(self) -> Dict[str, str]:
        """Get tool specifications as {name: description}."""
        return {name: tool.description for name, tool in self._tools.items()}

