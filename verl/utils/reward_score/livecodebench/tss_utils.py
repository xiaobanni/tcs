import re
from typing import Tuple, List, Any
import json


def process_response(response_text: str) -> Tuple[bool, List[Any]]:
    """
    Process a response text by extracting and parsing JSON content.

    Args:
        response_text: The text response from the model.

    Returns:
        Tuple containing:
        - Boolean indicating if processing was successful
        - List of parsed JSON objects (if successful) or empty list
    """
    # Match content between ```json {}```
    json_matches = re.findall(
        r'```json\s*(.*?)\s*```', response_text, re.DOTALL)

    if not json_matches:
        return False, []

    # Save all matched JSON content as a list
    json_contents = []
    success = False

    for json_match in json_matches:
        try:
            # First attempt to parse JSON
            json_obj = json.loads(json_match)
            json_contents.append(json_obj)
            success = True
        except json.JSONDecodeError:
            # First parse failed, try to trim until the last ']'
            try:
                # Find position of last ']'
                last_bracket_index = json_match.rfind(']')
                if last_bracket_index == -1:
                    # If no ']' found, raise error
                    continue
                # Trim until the last ']' (including ']')
                trimmed_json = json_match[:last_bracket_index + 1]
                # Second attempt to parse
                json_obj = json.loads(trimmed_json)
                json_contents.append(json_obj)
                success = True
            except json.JSONDecodeError:
                # Second parse also failed, ignore this match
                continue

    return success, json_contents


# Example cases
if __name__ == "__main__":
    # Example 1: Valid JSON response
    example_response1 = """Here's the solution:
```json
[
  {
    "input": "[6, 0, 3]\\n2",
    "output": "4"
  }
]
```
"""
    success1, json_contents1 = process_response(example_response1)
    print(f"Example 1 - Success: {success1}, Content: {json_contents1}")
