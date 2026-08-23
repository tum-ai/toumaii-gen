import json
from typing import Dict, List, Optional


class ToolParameter:
    """Represents a parameter of a tool function."""

    def __init__(
        self,
        name: str,
        param_type: str,
        description: str = "",
        required: bool = False,
        enum: Optional[List[str]] = None,
    ):
        self.name: str = name
        self.param_type: str = param_type
        self.description: str = description
        self.required: bool = required
        self.enum: Optional[List[str]] = enum

    def to_dict(self) -> Dict:
        d: Dict = {"title": self.name.capitalize(), "type": self.param_type}
        if self.description:
            d["description"] = self.description
        if self.enum:
            d["enum"] = self.enum
        return d


class ToolDefinition:
    """Represents a single tool/function that can be called by the model."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[List[ToolParameter]] = None,
    ):
        self.name: str = name
        self.description: str = description
        self.parameters: List[ToolParameter] = parameters or []

    @property
    def required_params(self) -> List[str]:
        return [p.name for p in self.parameters if p.required]

    def to_dict(self) -> Dict:
        properties = {}
        for param in self.parameters:
            properties[param.name] = param.to_dict()

        schema: Dict = {
            "type": "object",
            "properties": properties,
        }
        if self.required_params:
            schema["required"] = self.required_params
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ToolDefinition":
        """Parse a tool definition from the OpenAI-style dict format.

        Accepts both the wrapped form:
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        and the bare function form:
            {"name": ..., "description": ..., "parameters": {...}}
        """
        if "function" in d and isinstance(d.get("function"), dict):
            inner = d["function"]
        else:
            inner = d

        params_schema = inner.get("parameters", {})
        required_names: set = set(params_schema.get("required", []))
        param_list: List[ToolParameter] = []
        for pname, pmeta in params_schema.get("properties", {}).items():
            param_list.append(
                ToolParameter(
                    name=pname,
                    param_type=pmeta.get("type", "string"),
                    description=pmeta.get("description", ""),
                    required=pname in required_names,
                    enum=pmeta.get("enum"),
                )
            )

        return cls(
            name=inner["name"],
            description=inner.get("description", ""),
            parameters=param_list,
        )


class ToolCatalog:
    """
    Represents a collection of tool definitions available to the model.
    Converts to the format expected by tool_catalog in CanonicalFormat.
    """

    def __init__(self, tools: Optional[List[ToolDefinition]] = None):
        self.tools: List[ToolDefinition] = tools or []

    def add_tool(self, tool: ToolDefinition) -> None:
        self.tools.append(tool)

    def add_from_list(self, tool_dicts: List[Dict]) -> None:
        """Append tools parsed from a list of OpenAI-style tool dicts."""
        for d in tool_dicts:
            self.tools.append(ToolDefinition.from_dict(d))

    def to_list(self) -> List[Dict]:
        return [tool.to_dict() for tool in self.tools]

    @classmethod
    def from_list(cls, tool_dicts: List[Dict]) -> "ToolCatalog":
        """Create a catalog from a list of OpenAI-style tool dicts."""
        catalog = cls()
        catalog.add_from_list(tool_dicts)
        return catalog

    @classmethod
    def from_json(cls, json_str: str) -> "ToolCatalog":
        """Create a catalog from a JSON string (list of OpenAI-style tool dicts)."""
        data = json.loads(json_str)
        if not isinstance(data, list):
            raise ValueError("JSON must contain a list of tool definitions")
        return cls.from_list(data)

    @classmethod
    def from_example(cls) -> "ToolCatalog":
        """Create a catalog with the example weather functions."""
        catalog = cls()
        catalog.add_tool(
            ToolDefinition(
                name="weather-get_alerts",
                description="Get weather alerts for a US state.",
                parameters=[
                    ToolParameter(
                        name="state",
                        param_type="string",
                        description="Two-letter US state code (e.g. CA, NY)",
                        required=True,
                    ),
                ],
            )
        )
        catalog.add_tool(
            ToolDefinition(
                name="weather-get_forecast",
                description="Get weather forecast for a location.",
                parameters=[
                    ToolParameter(
                        name="latitude",
                        param_type="number",
                        description="Latitude of the location",
                        required=True,
                    ),
                    ToolParameter(
                        name="longitude",
                        param_type="number",
                        description="Longitude of the location",
                        required=True,
                    ),
                ],
            )
        )
        return catalog


class ToolCall:
    """Represents a single tool call made by the model."""

    def __init__(self, tool_name: str, arguments: Dict):
        self.tool_name: str = tool_name
        self.arguments: Dict = arguments

    def to_dict(self) -> Dict:
        # check if arguments is string, if so try to parse as json, if fails keep as string
        if isinstance(self.arguments, str):
            try:
                self.arguments = json.loads(self.arguments)
            except json.JSONDecodeError:
                pass
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        }

    @staticmethod
    def from_dict(d: Dict) -> "ToolCall":
        return ToolCall(tool_name=d["tool_name"], arguments=d["arguments"])
