"""
DRIFTBENCH Tool Schema Drift Tasks

Creates tasks that test RAG agents' ability to handle tool/API schema changes.
This is a novel contribution - no prior benchmark tests schema evolution.
"""

import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ToolDriftTask:
    """A task that tests handling of tool schema drift."""
    task_id: str
    question: str
    tool_name: str
    drift_type: str  # param_renamed, param_added, param_removed, unit_changed, type_changed
    schema_v1: Dict  # Old tool schema
    schema_v2: Dict  # New tool schema
    correct_call_v1: Dict  # Correct tool call under v1
    correct_call_v2: Dict  # Correct tool call under v2
    context_v1: str  # Documentation for v1
    context_v2: str  # Documentation for v2
    difficulty: str


# Tool schema drift examples (inspired by real API changes)
TOOL_DRIFT_EXAMPLES = [
    # Parameter renamed
    {
        "drift_type": "param_renamed",
        "tool_name": "search_database",
        "description": "Search parameter renamed from 'query' to 'search_term'",
        "schema_v1": {
            "name": "search_database",
            "parameters": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 10}
            }
        },
        "schema_v2": {
            "name": "search_database",
            "parameters": {
                "search_term": {"type": "string", "description": "Search term to look for"},
                "limit": {"type": "integer", "default": 10}
            }
        },
        "question": "How do I search the database for 'machine learning' with a limit of 5 results?",
        "correct_call_v1": {"tool": "search_database", "args": {"query": "machine learning", "limit": 5}},
        "correct_call_v2": {"tool": "search_database", "args": {"search_term": "machine learning", "limit": 5}},
        "difficulty": "easy"
    },
    # Parameter added (required)
    {
        "drift_type": "param_added",
        "tool_name": "create_user",
        "description": "Required 'email' parameter added to user creation",
        "schema_v1": {
            "name": "create_user",
            "parameters": {
                "username": {"type": "string", "required": True},
                "password": {"type": "string", "required": True}
            }
        },
        "schema_v2": {
            "name": "create_user",
            "parameters": {
                "username": {"type": "string", "required": True},
                "password": {"type": "string", "required": True},
                "email": {"type": "string", "required": True, "description": "User email for verification"}
            }
        },
        "question": "Create a new user with username 'john_doe' and password 'secret123'",
        "correct_call_v1": {"tool": "create_user", "args": {"username": "john_doe", "password": "secret123"}},
        "correct_call_v2": {"tool": "create_user", "args": {"username": "john_doe", "password": "secret123", "email": "required_field"}},
        "difficulty": "medium"
    },
    # Unit changed (meters to feet)
    {
        "drift_type": "unit_changed",
        "tool_name": "calculate_area",
        "description": "Input units changed from meters to feet",
        "schema_v1": {
            "name": "calculate_area",
            "parameters": {
                "length": {"type": "number", "description": "Length in meters"},
                "width": {"type": "number", "description": "Width in meters"}
            }
        },
        "schema_v2": {
            "name": "calculate_area",
            "parameters": {
                "length": {"type": "number", "description": "Length in feet"},
                "width": {"type": "number", "description": "Width in feet"}
            }
        },
        "question": "Calculate the area of a room that is 3 meters by 4 meters",
        "correct_call_v1": {"tool": "calculate_area", "args": {"length": 3, "width": 4}},
        "correct_call_v2": {"tool": "calculate_area", "args": {"length": 9.84, "width": 13.12}},  # meters to feet
        "difficulty": "hard"
    },
    # Type changed (string to enum)
    {
        "drift_type": "type_changed",
        "tool_name": "set_priority",
        "description": "Priority changed from free string to enum",
        "schema_v1": {
            "name": "set_priority",
            "parameters": {
                "task_id": {"type": "string"},
                "priority": {"type": "string", "description": "Priority level (any string)"}
            }
        },
        "schema_v2": {
            "name": "set_priority",
            "parameters": {
                "task_id": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
            }
        },
        "question": "Set task ABC123 priority to 'urgent'",
        "correct_call_v1": {"tool": "set_priority", "args": {"task_id": "ABC123", "priority": "urgent"}},
        "correct_call_v2": {"tool": "set_priority", "args": {"task_id": "ABC123", "priority": "critical"}},  # Must map to enum
        "difficulty": "medium"
    },
    # Return type changed
    {
        "drift_type": "return_changed",
        "tool_name": "get_user",
        "description": "Return type changed from flat object to nested object",
        "schema_v1": {
            "name": "get_user",
            "parameters": {"user_id": {"type": "string"}},
            "returns": {"name": "string", "email": "string"}
        },
        "schema_v2": {
            "name": "get_user",
            "parameters": {"user_id": {"type": "string"}},
            "returns": {"user": {"name": "string", "email": "string"}, "metadata": {"created_at": "string"}}
        },
        "question": "Get the email of user with ID 12345",
        "correct_call_v1": {"tool": "get_user", "args": {"user_id": "12345"}, "extract": "result.email"},
        "correct_call_v2": {"tool": "get_user", "args": {"user_id": "12345"}, "extract": "result.user.email"},
        "difficulty": "hard"
    },
    # Default value changed
    {
        "drift_type": "default_changed",
        "tool_name": "list_files",
        "description": "Default sort order changed from ascending to descending",
        "schema_v1": {
            "name": "list_files",
            "parameters": {
                "directory": {"type": "string"},
                "sort_order": {"type": "string", "default": "asc", "enum": ["asc", "desc"]}
            }
        },
        "schema_v2": {
            "name": "list_files",
            "parameters": {
                "directory": {"type": "string"},
                "sort_order": {"type": "string", "default": "desc", "enum": ["asc", "desc"]}
            }
        },
        "question": "List files in /home/user sorted in ascending order",
        "correct_call_v1": {"tool": "list_files", "args": {"directory": "/home/user"}},  # asc is default
        "correct_call_v2": {"tool": "list_files", "args": {"directory": "/home/user", "sort_order": "asc"}},  # must specify
        "difficulty": "medium"
    },
    # Parameter removed (now inferred)
    {
        "drift_type": "param_removed",
        "tool_name": "send_notification",
        "description": "Channel parameter removed, now auto-detected from content",
        "schema_v1": {
            "name": "send_notification",
            "parameters": {
                "message": {"type": "string"},
                "channel": {"type": "string", "enum": ["email", "sms", "push"]}
            }
        },
        "schema_v2": {
            "name": "send_notification",
            "parameters": {
                "message": {"type": "string", "description": "Message (channel auto-detected)"}
            }
        },
        "question": "Send a push notification saying 'Meeting in 5 minutes'",
        "correct_call_v1": {"tool": "send_notification", "args": {"message": "Meeting in 5 minutes", "channel": "push"}},
        "correct_call_v2": {"tool": "send_notification", "args": {"message": "Meeting in 5 minutes"}},
        "difficulty": "easy"
    },
    # Endpoint renamed
    {
        "drift_type": "tool_renamed",
        "tool_name": "get_weather",
        "description": "Tool renamed from get_weather to fetch_weather_data",
        "schema_v1": {
            "name": "get_weather",
            "parameters": {"city": {"type": "string"}}
        },
        "schema_v2": {
            "name": "fetch_weather_data",
            "parameters": {"city": {"type": "string"}}
        },
        "question": "What's the weather in New York?",
        "correct_call_v1": {"tool": "get_weather", "args": {"city": "New York"}},
        "correct_call_v2": {"tool": "fetch_weather_data", "args": {"city": "New York"}},
        "difficulty": "medium"
    },
    # Rate limit added
    {
        "drift_type": "constraint_added",
        "tool_name": "batch_process",
        "description": "Maximum batch size reduced from 1000 to 100",
        "schema_v1": {
            "name": "batch_process",
            "parameters": {
                "items": {"type": "array", "maxItems": 1000}
            }
        },
        "schema_v2": {
            "name": "batch_process",
            "parameters": {
                "items": {"type": "array", "maxItems": 100, "description": "Max 100 items per batch"}
            }
        },
        "question": "Process 500 items in the batch processor",
        "correct_call_v1": {"tool": "batch_process", "args": {"items": "[500 items]"}, "valid": True},
        "correct_call_v2": {"tool": "batch_process", "args": {"items": "[100 items x 5 batches]"}, "valid": True, "note": "Must split into 5 batches"},
        "difficulty": "hard"
    },
    # Authentication changed
    {
        "drift_type": "auth_changed",
        "tool_name": "access_resource",
        "description": "Auth changed from API key to OAuth token",
        "schema_v1": {
            "name": "access_resource",
            "parameters": {
                "resource_id": {"type": "string"},
                "api_key": {"type": "string", "description": "API key for authentication"}
            }
        },
        "schema_v2": {
            "name": "access_resource",
            "parameters": {
                "resource_id": {"type": "string"},
                "bearer_token": {"type": "string", "description": "OAuth2 bearer token"}
            }
        },
        "question": "Access resource RES001 with my API key 'abc123'",
        "correct_call_v1": {"tool": "access_resource", "args": {"resource_id": "RES001", "api_key": "abc123"}},
        "correct_call_v2": {"tool": "access_resource", "args": {"resource_id": "RES001", "bearer_token": "oauth_token_here"}, "note": "Must use OAuth token, not API key"},
        "difficulty": "hard"
    },
]


def generate_tool_drift_tasks() -> List[ToolDriftTask]:
    """Generate tool drift tasks from examples."""
    tasks = []

    for i, example in enumerate(TOOL_DRIFT_EXAMPLES):
        task = ToolDriftTask(
            task_id=f"tool_drift_{i:04d}",
            question=example["question"],
            tool_name=example["tool_name"],
            drift_type=example["drift_type"],
            schema_v1=example["schema_v1"],
            schema_v2=example["schema_v2"],
            correct_call_v1=example["correct_call_v1"],
            correct_call_v2=example["correct_call_v2"],
            context_v1=f"[API v1] {example['tool_name']}: {json.dumps(example['schema_v1'], indent=2)}",
            context_v2=f"[API v2] {example['tool_name']}: {json.dumps(example['schema_v2'], indent=2)}\n\nNote: {example['description']}",
            difficulty=example["difficulty"]
        )
        tasks.append(task)

    return tasks


def save_tool_drift_dataset(output_dir: Path):
    """Save tool drift tasks to JSON."""
    tasks = generate_tool_drift_tasks()

    dataset = {
        "name": "DRIFTBENCH-ToolDrift",
        "version": "0.1",
        "created": datetime.now().isoformat(),
        "n_tasks": len(tasks),
        "drift_types": {
            "param_renamed": len([t for t in tasks if t.drift_type == "param_renamed"]),
            "param_added": len([t for t in tasks if t.drift_type == "param_added"]),
            "param_removed": len([t for t in tasks if t.drift_type == "param_removed"]),
            "unit_changed": len([t for t in tasks if t.drift_type == "unit_changed"]),
            "type_changed": len([t for t in tasks if t.drift_type == "type_changed"]),
            "default_changed": len([t for t in tasks if t.drift_type == "default_changed"]),
            "return_changed": len([t for t in tasks if t.drift_type == "return_changed"]),
            "tool_renamed": len([t for t in tasks if t.drift_type == "tool_renamed"]),
            "constraint_added": len([t for t in tasks if t.drift_type == "constraint_added"]),
            "auth_changed": len([t for t in tasks if t.drift_type == "auth_changed"]),
        },
        "tasks": [asdict(t) for t in tasks]
    }

    output_path = output_dir / "tool_drift_tasks.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(tasks)} tool drift tasks to {output_path}")
    return output_path


def main():
    """Generate tool drift dataset."""
    print("=" * 60)
    print("  DRIFTBENCH - Tool Schema Drift Tasks")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    save_tool_drift_dataset(output_dir)

    # Print summary
    tasks = generate_tool_drift_tasks()
    print(f"\n  Generated {len(tasks)} tool drift tasks:")
    for drift_type in set(t.drift_type for t in tasks):
        count = len([t for t in tasks if t.drift_type == drift_type])
        print(f"    {drift_type}: {count}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
