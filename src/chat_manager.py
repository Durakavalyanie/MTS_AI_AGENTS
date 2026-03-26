from __future__ import annotations

import json
import re
from typing import Any

CODE_AUTHORS = {"DataAnalyst", "DataEngineer", "MLEngineer"}
VALID_AGENT_NAMES = {
    "Orchestrator",
    "DataAnalyst",
    "DataEngineer",
    "MLEngineer",
    "CodeExecutor",
}


def _extract_tool_parts(part: str) -> tuple[str | None, str, str | None]:
    part = part.strip()
    
    code_match = re.search(r"```(?:python)?\n(.*?)```", part, re.DOTALL | re.IGNORECASE)
    code = code_match.group(1).strip() if code_match else None
    
    json_part = part[:code_match.start()] if code_match else part
    json_part = json_part.strip()
    
    name_match = re.match(r"^([a-zA-Z0-9_]+)", json_part)
    if not name_match:
        return None, "", code
        
    tool_name = name_match.group(1)
    
    json_match = re.search(r"(\{.*\})", json_part, re.DOTALL)
    json_str = json_match.group(1) if json_match else "{}"
    
    return tool_name, json_str, code


def validate_tool_calls(content: str, role: str) -> list[str]:
    errors = []
    if "[TOOL_CALL]" not in content:
        return ["Error: No [TOOL_CALL] found in your message. You must use exactly one tool and format it correctly."]
    
    parts = content.split("[TOOL_CALL]")
    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
            
        tool_name, json_str, code = _extract_tool_parts(part)
        if not tool_name:
            errors.append("Error: Could not parse tool name. Format must be [TOOL_CALL]tool_name\\n{...}")
            continue
            
        # Check role permissions
        if role == "Orchestrator" and tool_name not in {"delegate", "submit_to_kaggle"}:
            errors.append(f"Error: Orchestrator can only use 'delegate' or 'submit_to_kaggle' tool, got '{tool_name}'.")
        elif role in {"DataAnalyst", "DataEngineer", "MLEngineer"} and tool_name not in {"execute_code", "send_message"}:
            errors.append(f"Error: {role} can only use 'execute_code' or 'send_message', got '{tool_name}'.")
            
        try:
            args = json.loads(json_str, strict=False) if json_str else {}
            if not isinstance(args, dict):
                errors.append(f"Error: Tool arguments for '{tool_name}' must be a JSON object (dictionary).")
                continue
                
            if tool_name == "delegate":
                for req in ["thoughts", "directive", "next_speaker"]:
                    if req not in args:
                        errors.append(f"Error: Missing required JSON argument '{req}' for tool '{tool_name}'.")
                if args.get("next_speaker") not in {"DataAnalyst", "DataEngineer", "MLEngineer"}:
                    errors.append("Error: 'next_speaker' must be one of: DataAnalyst, DataEngineer, MLEngineer.")
                    
            elif tool_name == "submit_to_kaggle":
                for req in ["thoughts", "message"]:
                    if req not in args:
                        errors.append(f"Error: Missing required JSON argument '{req}' for tool '{tool_name}'.")

            elif tool_name == "execute_code":
                for req in ["thoughts", "expected_outcome"]:
                    if req not in args:
                        errors.append(f"Error: Missing required JSON argument '{req}' for tool '{tool_name}'.")
                if not code:
                    errors.append(f"Error: Missing markdown code block ```python ... ``` for tool '{tool_name}'. Code must be placed outside the JSON.")
                    
            elif tool_name == "send_message":
                for req in ["thoughts", "message"]:
                    if req not in args:
                        errors.append(f"Error: Missing required JSON argument '{req}' for tool '{tool_name}'.")
                        
        except json.JSONDecodeError as e:
            errors.append(f"Error: Invalid JSON format for tool '{tool_name}'. JSONDecodeError: {str(e)}. Make sure your JSON is valid.")
            
    return errors


def parse_tool_calls(content: str) -> list[tuple[str, dict[str, Any]]]:
    calls = []
    parts = content.split("[TOOL_CALL]")
    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
        
        tool_name, json_str, code = _extract_tool_parts(part)
        if not tool_name:
            continue
            
        try:
            args = json.loads(json_str, strict=False) if json_str else {}
            if isinstance(args, dict):
                if code:
                    args["code"] = code
                calls.append((tool_name, args))
        except json.JSONDecodeError:
            pass
    return calls
