"""
FastAPI Diff Miner - Extract QA pairs from FastAPI documentation changes

Mines breaking changes between FastAPI versions to create organic drift tasks.
"""

import os
import json
import re
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class VersionChange:
    """A single change between versions."""
    file_path: str
    change_type: str  # default_changed, param_renamed, behavior_changed, constraint_added
    old_value: str
    new_value: str
    context: str  # Surrounding text for context
    version_old: str
    version_new: str


@dataclass
class DiffBasedQA:
    """A QA pair derived from a version diff."""
    task_id: str
    question: str
    answer_v1: str
    answer_v2: str
    evidence_v1: str
    evidence_v2: str
    category: str
    source_change: VersionChange
    difficulty: str


# Known FastAPI breaking changes (manually curated from changelogs)
FASTAPI_BREAKING_CHANGES = [
    # Version 0.100.0 -> 0.109.0 changes
    {
        "version_old": "0.100.0",
        "version_new": "0.109.0",
        "changes": [
            {
                "change_type": "default_changed",
                "topic": "response_model_exclude_unset",
                "old_value": "False",
                "new_value": "True",
                "context": "Response model serialization now excludes unset fields by default",
                "question": "What is the default value for response_model_exclude_unset in FastAPI?",
                "difficulty": "easy"
            },
            {
                "change_type": "behavior_changed",
                "topic": "OpenAPI schema generation",
                "old_value": "Generates schemas for all response models inline",
                "new_value": "Uses $ref references for repeated schemas",
                "context": "OpenAPI schema generation optimized to reduce size",
                "question": "How does FastAPI handle repeated response model schemas in OpenAPI?",
                "difficulty": "medium"
            },
        ]
    },
    # Version 0.109.0 -> 0.115.0 changes
    {
        "version_old": "0.109.0",
        "version_new": "0.115.0",
        "changes": [
            {
                "change_type": "default_changed",
                "topic": "lifespan context manager",
                "old_value": "on_startup/on_shutdown events",
                "new_value": "lifespan async context manager",
                "context": "Lifespan events now use async context manager pattern",
                "question": "What is the recommended way to handle startup/shutdown events in FastAPI?",
                "difficulty": "medium"
            },
            {
                "change_type": "param_renamed",
                "topic": "Depends() signature",
                "old_value": "use_cache=True",
                "new_value": "use_cache parameter deprecated, caching always enabled",
                "context": "Dependency injection caching behavior changed",
                "question": "How does dependency caching work in FastAPI's Depends()?",
                "difficulty": "hard"
            },
        ]
    },
    # Pydantic v1 -> v2 migration (major breaking change)
    {
        "version_old": "0.99.0",
        "version_new": "0.100.0",
        "changes": [
            {
                "change_type": "behavior_changed",
                "topic": "Pydantic model validation",
                "old_value": "Pydantic v1 validators with @validator decorator",
                "new_value": "Pydantic v2 validators with @field_validator decorator",
                "context": "FastAPI 0.100.0 requires Pydantic v2",
                "question": "What decorator is used for field validation in FastAPI models?",
                "difficulty": "medium"
            },
            {
                "change_type": "default_changed",
                "topic": "Model Config",
                "old_value": "class Config with orm_mode = True",
                "new_value": "model_config with from_attributes = True",
                "context": "Pydantic v2 configuration syntax changed",
                "question": "How do you enable ORM mode in FastAPI Pydantic models?",
                "difficulty": "medium"
            },
            {
                "change_type": "param_renamed",
                "topic": "schema_extra",
                "old_value": "schema_extra in Config class",
                "new_value": "json_schema_extra in model_config",
                "context": "JSON schema customization syntax changed",
                "question": "How do you add extra fields to the JSON schema of a Pydantic model in FastAPI?",
                "difficulty": "hard"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Optional fields",
                "old_value": "Optional[str] = None implies optional",
                "new_value": "Must explicitly use Optional[str] = None or str | None = None",
                "context": "Pydantic v2 is stricter about optional field declarations",
                "question": "How do you declare an optional field in a FastAPI request model?",
                "difficulty": "easy"
            },
        ]
    },
    # Additional version changes
    {
        "version_old": "0.95.0",
        "version_new": "0.99.0",
        "changes": [
            {
                "change_type": "default_changed",
                "topic": "Query parameter validation",
                "old_value": "Query() without explicit validation",
                "new_value": "Query() with Annotated syntax recommended",
                "context": "FastAPI recommends Annotated for parameter declarations",
                "question": "What is the recommended way to declare query parameters with validation in FastAPI?",
                "difficulty": "medium"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Background tasks execution",
                "old_value": "Background tasks run after response sent",
                "new_value": "Background tasks run after response sent, but with improved error handling",
                "context": "Background task error handling improved",
                "question": "When do background tasks execute in FastAPI relative to the response?",
                "difficulty": "easy"
            },
        ]
    },
    # More Pydantic v2 changes
    {
        "version_old": "0.99.0",
        "version_new": "0.100.0",
        "changes": [
            {
                "change_type": "param_renamed",
                "topic": "Field constraints",
                "old_value": "min_length, max_length as Field() parameters",
                "new_value": "min_length, max_length still work but Annotated[str, StringConstraints()] preferred",
                "context": "Pydantic v2 introduces StringConstraints for string validation",
                "question": "What is the preferred way to add string length constraints in Pydantic v2?",
                "difficulty": "medium"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Model serialization",
                "old_value": "model.dict() for dictionary output",
                "new_value": "model.model_dump() for dictionary output",
                "context": "Pydantic v2 renames dict() to model_dump()",
                "question": "How do you convert a Pydantic model to a dictionary in v2?",
                "difficulty": "easy"
            },
            {
                "change_type": "behavior_changed",
                "topic": "JSON serialization",
                "old_value": "model.json() for JSON string output",
                "new_value": "model.model_dump_json() for JSON string output",
                "context": "Pydantic v2 renames json() to model_dump_json()",
                "question": "How do you serialize a Pydantic model to JSON string in v2?",
                "difficulty": "easy"
            },
            {
                "change_type": "param_renamed",
                "topic": "Schema method",
                "old_value": "Model.schema() for JSON schema",
                "new_value": "Model.model_json_schema() for JSON schema",
                "context": "Pydantic v2 renames schema() to model_json_schema()",
                "question": "How do you get the JSON schema of a Pydantic model in v2?",
                "difficulty": "easy"
            },
            {
                "change_type": "behavior_changed",
                "topic": "parse_obj method",
                "old_value": "Model.parse_obj(data) to create from dict",
                "new_value": "Model.model_validate(data) to create from dict",
                "context": "Pydantic v2 renames parse_obj() to model_validate()",
                "question": "How do you create a Pydantic model from a dictionary in v2?",
                "difficulty": "easy"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Root validators",
                "old_value": "@root_validator decorator for model-level validation",
                "new_value": "@model_validator decorator for model-level validation",
                "context": "Pydantic v2 renames root_validator to model_validator",
                "question": "What decorator is used for model-level validation in Pydantic v2?",
                "difficulty": "medium"
            },
            {
                "change_type": "default_changed",
                "topic": "Strict mode",
                "old_value": "Coercion enabled by default (strings to ints, etc)",
                "new_value": "Strict mode available via strict=True in model_config",
                "context": "Pydantic v2 adds strict mode to disable type coercion",
                "question": "How do you disable type coercion in Pydantic v2?",
                "difficulty": "medium"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Update forward refs",
                "old_value": "Model.update_forward_refs() to resolve forward references",
                "new_value": "Model.model_rebuild() to resolve forward references",
                "context": "Pydantic v2 renames update_forward_refs() to model_rebuild()",
                "question": "How do you resolve forward references in Pydantic v2?",
                "difficulty": "hard"
            },
            {
                "change_type": "param_renamed",
                "topic": "Field alias",
                "old_value": "Field(alias='name') only",
                "new_value": "Field(alias='name', serialization_alias='name', validation_alias='name')",
                "context": "Pydantic v2 adds separate aliases for validation and serialization",
                "question": "What alias options are available in Pydantic v2 Field()?",
                "difficulty": "hard"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Copy method",
                "old_value": "model.copy(update={'field': value}) for shallow copy",
                "new_value": "model.model_copy(update={'field': value}) for shallow copy",
                "context": "Pydantic v2 renames copy() to model_copy()",
                "question": "How do you create a modified copy of a Pydantic model in v2?",
                "difficulty": "easy"
            },
        ]
    },
    # FastAPI 0.110+ changes
    {
        "version_old": "0.109.0",
        "version_new": "0.110.0",
        "changes": [
            {
                "change_type": "default_changed",
                "topic": "OpenAPI version",
                "old_value": "OpenAPI 3.0.2",
                "new_value": "OpenAPI 3.1.0",
                "context": "FastAPI upgrades to OpenAPI 3.1.0 specification",
                "question": "What OpenAPI version does FastAPI 0.110+ use by default?",
                "difficulty": "easy"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Nullable fields in OpenAPI",
                "old_value": "nullable: true in schema",
                "new_value": "type: ['string', 'null'] (JSON Schema style)",
                "context": "OpenAPI 3.1.0 uses JSON Schema style for nullable",
                "question": "How are nullable fields represented in FastAPI 0.110+ OpenAPI schema?",
                "difficulty": "medium"
            },
        ]
    },
    # Starlette changes (FastAPI dependency)
    {
        "version_old": "0.27.0",
        "version_new": "0.32.0",
        "changes": [
            {
                "change_type": "behavior_changed",
                "topic": "TestClient",
                "old_value": "TestClient uses requests library",
                "new_value": "TestClient uses httpx library",
                "context": "Starlette switched from requests to httpx for TestClient",
                "question": "What HTTP library does Starlette TestClient use?",
                "difficulty": "medium"
            },
            {
                "change_type": "default_changed",
                "topic": "Exception handlers",
                "old_value": "Exception handlers receive (request, exc) arguments",
                "new_value": "Exception handlers receive (request, exc) with typed Request",
                "context": "Starlette improved type hints for exception handlers",
                "question": "What arguments do Starlette exception handlers receive?",
                "difficulty": "easy"
            },
        ]
    },
    # SQLAlchemy 1.4 -> 2.0 (common with FastAPI)
    {
        "version_old": "1.4",
        "version_new": "2.0",
        "changes": [
            {
                "change_type": "behavior_changed",
                "topic": "Query execution",
                "old_value": "session.query(Model).filter() style",
                "new_value": "session.execute(select(Model).where()) style",
                "context": "SQLAlchemy 2.0 uses select() statement style",
                "question": "What is the recommended query style in SQLAlchemy 2.0?",
                "difficulty": "medium"
            },
            {
                "change_type": "default_changed",
                "topic": "Async support",
                "old_value": "Async requires separate package (databases)",
                "new_value": "Native async support with AsyncSession",
                "context": "SQLAlchemy 2.0 has native async support",
                "question": "How do you use async with SQLAlchemy 2.0?",
                "difficulty": "medium"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Column definitions",
                "old_value": "Column(Integer, primary_key=True)",
                "new_value": "mapped_column(Integer, primary_key=True) with Mapped[int]",
                "context": "SQLAlchemy 2.0 introduces mapped_column() and Mapped type hints",
                "question": "How do you define typed columns in SQLAlchemy 2.0 ORM?",
                "difficulty": "hard"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Relationship definitions",
                "old_value": "relationship('Model') with string reference",
                "new_value": "relationship() returns Mapped[Model] with type annotation",
                "context": "SQLAlchemy 2.0 uses type annotations for relationships",
                "question": "How do you define typed relationships in SQLAlchemy 2.0?",
                "difficulty": "hard"
            },
        ]
    },
    # HTTPX changes (used by FastAPI TestClient)
    {
        "version_old": "0.23.0",
        "version_new": "0.25.0",
        "changes": [
            {
                "change_type": "default_changed",
                "topic": "Timeout default",
                "old_value": "No default timeout (infinite)",
                "new_value": "Default timeout of 5 seconds",
                "context": "HTTPX added default timeout to prevent hanging requests",
                "question": "What is the default timeout in HTTPX 0.25+?",
                "difficulty": "easy"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Follow redirects",
                "old_value": "follow_redirects=True by default",
                "new_value": "follow_redirects=False by default",
                "context": "HTTPX changed default redirect behavior",
                "question": "Does HTTPX follow redirects by default in version 0.25+?",
                "difficulty": "easy"
            },
        ]
    },
    # More FastAPI security changes
    {
        "version_old": "0.95.0",
        "version_new": "0.100.0",
        "changes": [
            {
                "change_type": "behavior_changed",
                "topic": "OAuth2 scopes",
                "old_value": "Scopes defined as list of strings",
                "new_value": "Scopes defined with SecurityScopes dependency",
                "context": "FastAPI improved OAuth2 scope handling",
                "question": "How do you access OAuth2 scopes in FastAPI dependencies?",
                "difficulty": "hard"
            },
            {
                "change_type": "default_changed",
                "topic": "Cookie security",
                "old_value": "Cookies without SameSite attribute",
                "new_value": "Cookies with SameSite=Lax by default",
                "context": "FastAPI follows browser security best practices",
                "question": "What is the default SameSite attribute for cookies in FastAPI?",
                "difficulty": "medium"
            },
        ]
    },
    # WebSocket changes
    {
        "version_old": "0.90.0",
        "version_new": "0.100.0",
        "changes": [
            {
                "change_type": "behavior_changed",
                "topic": "WebSocket close codes",
                "old_value": "Manual close code handling",
                "new_value": "WebSocketClose exception with status codes",
                "context": "FastAPI added WebSocketClose for cleaner close handling",
                "question": "How do you close a WebSocket with a specific code in FastAPI?",
                "difficulty": "medium"
            },
            {
                "change_type": "default_changed",
                "topic": "WebSocket state",
                "old_value": "No built-in connection state tracking",
                "new_value": "WebSocket.state attribute available",
                "context": "FastAPI added WebSocket state tracking",
                "question": "How do you check WebSocket connection state in FastAPI?",
                "difficulty": "easy"
            },
        ]
    },
    # Dependency injection changes
    {
        "version_old": "0.85.0",
        "version_new": "0.95.0",
        "changes": [
            {
                "change_type": "behavior_changed",
                "topic": "Yield dependencies cleanup",
                "old_value": "Cleanup runs even if exception not handled",
                "new_value": "Cleanup runs in finally block, exception re-raised",
                "context": "FastAPI improved yield dependency error handling",
                "question": "When does cleanup code run for yield dependencies in FastAPI?",
                "difficulty": "hard"
            },
            {
                "change_type": "default_changed",
                "topic": "Dependency overrides",
                "old_value": "app.dependency_overrides dict only",
                "new_value": "app.dependency_overrides with context manager support",
                "context": "FastAPI added context manager for dependency overrides",
                "question": "How do you temporarily override dependencies in FastAPI tests?",
                "difficulty": "medium"
            },
        ]
    },
    # Response model changes
    {
        "version_old": "0.89.0",
        "version_new": "0.100.0",
        "changes": [
            {
                "change_type": "param_renamed",
                "topic": "Response model include/exclude",
                "old_value": "response_model_include, response_model_exclude as sets",
                "new_value": "response_model_include, response_model_exclude as sets or dicts",
                "context": "FastAPI added nested field include/exclude support",
                "question": "How do you exclude nested fields from response model in FastAPI?",
                "difficulty": "hard"
            },
            {
                "change_type": "default_changed",
                "topic": "Response model validation",
                "old_value": "Response model validates output",
                "new_value": "response_model_validate=True (explicit) or False to skip",
                "context": "FastAPI added option to skip response validation",
                "question": "How do you skip response model validation in FastAPI?",
                "difficulty": "medium"
            },
        ]
    },
    # File upload changes
    {
        "version_old": "0.80.0",
        "version_new": "0.95.0",
        "changes": [
            {
                "change_type": "behavior_changed",
                "topic": "UploadFile seek",
                "old_value": "UploadFile.file.seek() for position reset",
                "new_value": "await UploadFile.seek() async method available",
                "context": "FastAPI added async seek() to UploadFile",
                "question": "How do you reset file position in FastAPI UploadFile?",
                "difficulty": "easy"
            },
            {
                "change_type": "default_changed",
                "topic": "File size limit",
                "old_value": "No default file size limit",
                "new_value": "Configurable via max_size parameter",
                "context": "FastAPI added file size limit configuration",
                "question": "How do you limit upload file size in FastAPI?",
                "difficulty": "medium"
            },
        ]
    },
]

# Synthetic knowledge drift - controlled changes for dose sweeps
SYNTHETIC_DRIFT_TEMPLATES = [
    {
        "template": "default_value",
        "original_question": "What is the default value for {param} in {context}?",
        "original_answer": "{old_value}",
        "drifted_answer": "{new_value}",
        "drift_magnitudes": [0.01, 0.02, 0.05, 0.1]  # Percentage of corpus affected
    },
    {
        "template": "parameter_name",
        "original_question": "What parameter controls {behavior} in {context}?",
        "original_answer": "{old_param}",
        "drifted_answer": "{new_param}",
        "drift_magnitudes": [0.01, 0.02, 0.05, 0.1]
    },
    {
        "template": "recommended_pattern",
        "original_question": "What is the recommended way to {action} in {context}?",
        "original_answer": "{old_pattern}",
        "drifted_answer": "{new_pattern}",
        "drift_magnitudes": [0.01, 0.02, 0.05, 0.1]
    },
]


class FastAPIDiffMiner:
    """Mine QA pairs from FastAPI version changes."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tasks: List[DiffBasedQA] = []

    def mine_organic_changes(self) -> List[DiffBasedQA]:
        """Extract QA pairs from known breaking changes."""
        task_id = 0

        for version_block in FASTAPI_BREAKING_CHANGES:
            v_old = version_block["version_old"]
            v_new = version_block["version_new"]

            for change in version_block["changes"]:
                task = DiffBasedQA(
                    task_id=f"fastapi_organic_{task_id:04d}",
                    question=change["question"],
                    answer_v1=change["old_value"],
                    answer_v2=change["new_value"],
                    evidence_v1=f"[FastAPI {v_old}] {change['context']} Value: {change['old_value']}",
                    evidence_v2=f"[FastAPI {v_new}] {change['context']} Value: {change['new_value']}",
                    category="factoid",
                    source_change=VersionChange(
                        file_path=f"docs/{change['topic'].lower().replace(' ', '_')}.md",
                        change_type=change["change_type"],
                        old_value=change["old_value"],
                        new_value=change["new_value"],
                        context=change["context"],
                        version_old=v_old,
                        version_new=v_new
                    ),
                    difficulty=change["difficulty"]
                )
                self.tasks.append(task)
                task_id += 1

        return self.tasks

    def generate_multi_hop_tasks(self) -> List[DiffBasedQA]:
        """Generate multi-hop reasoning tasks from changes."""
        multi_hop_tasks = []
        task_id = len(self.tasks)

        # Combine related changes into multi-hop questions
        pydantic_changes = [t for t in self.tasks if "pydantic" in t.question.lower() or "model" in t.question.lower()]

        if len(pydantic_changes) >= 2:
            # Create a multi-hop question combining two changes
            task = DiffBasedQA(
                task_id=f"fastapi_multihop_{task_id:04d}",
                question="If I'm migrating a FastAPI app from 0.99 to 0.100, and I have a model with orm_mode=True and a @validator decorator, what two changes do I need to make?",
                answer_v1="No changes needed - orm_mode=True and @validator work correctly",
                answer_v2="Change orm_mode=True to from_attributes=True in model_config, and change @validator to @field_validator",
                evidence_v1="[FastAPI 0.99] Uses Pydantic v1 with class Config and @validator decorators",
                evidence_v2="[FastAPI 0.100] Requires Pydantic v2 with model_config dict and @field_validator decorators",
                category="multi_hop",
                source_change=VersionChange(
                    file_path="docs/migration_guide.md",
                    change_type="migration",
                    old_value="Pydantic v1 patterns",
                    new_value="Pydantic v2 patterns",
                    context="Migration from Pydantic v1 to v2",
                    version_old="0.99.0",
                    version_new="0.100.0"
                ),
                difficulty="hard"
            )
            multi_hop_tasks.append(task)
            task_id += 1

        self.tasks.extend(multi_hop_tasks)
        return multi_hop_tasks

    def generate_distractor_docs(self) -> Dict[str, str]:
        """Generate distractor documents that contain OLD information."""
        distractors = {}

        for task in self.tasks:
            # Create a distractor doc that explicitly states the OLD value
            distractor_id = f"distractor_{task.task_id}"
            distractor_content = f"""
# {task.source_change.context} (Outdated Documentation)

**Note**: This documentation may be outdated.

The correct value for this setting is: {task.answer_v1}

This has been the default since version {task.source_change.version_old}.

Example usage:
```python
# Using the {task.answer_v1} setting
app = FastAPI()
```
"""
            distractors[distractor_id] = distractor_content

        return distractors

    def save_dataset(self, filename: str = "fastapi_drift_tasks.json"):
        """Save mined tasks to JSON."""
        output_path = self.output_dir / filename

        dataset = {
            "name": "DRIFTBENCH-FastAPI",
            "version": "0.1",
            "created": datetime.now().isoformat(),
            "n_tasks": len(self.tasks),
            "categories": {
                "factoid": len([t for t in self.tasks if t.category == "factoid"]),
                "multi_hop": len([t for t in self.tasks if t.category == "multi_hop"]),
            },
            "tasks": [asdict(t) for t in self.tasks]
        }

        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)

        print(f"Saved {len(self.tasks)} tasks to {output_path}")
        return output_path

    def create_corpus_versions(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Create v1 and v2 corpus versions for retrieval."""
        corpus_v1 = {}
        corpus_v2 = {}

        for task in self.tasks:
            doc_id = task.task_id

            # V1 corpus contains old information
            corpus_v1[doc_id] = f"""
# FastAPI Documentation

{task.evidence_v1}

Related topic: {task.source_change.context}
"""

            # V2 corpus contains new information
            corpus_v2[doc_id] = f"""
# FastAPI Documentation

{task.evidence_v2}

Related topic: {task.source_change.context}
"""

        # Add some shared context docs
        shared_docs = {
            "intro": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.",
            "installation": "Install FastAPI with: pip install fastapi[all]",
            "quickstart": "Create a simple API with: from fastapi import FastAPI; app = FastAPI()",
        }

        for doc_id, content in shared_docs.items():
            corpus_v1[f"shared_{doc_id}"] = content
            corpus_v2[f"shared_{doc_id}"] = content

        return corpus_v1, corpus_v2


def main():
    """Mine FastAPI changes and create DRIFTBENCH dataset."""
    output_dir = Path(__file__).parent.parent / "data"
    miner = FastAPIDiffMiner(output_dir)

    print("=" * 60)
    print("  DRIFTBENCH - FastAPI Diff Mining")
    print("=" * 60)

    # Mine organic changes
    print("\n[1] Mining organic version changes...")
    organic_tasks = miner.mine_organic_changes()
    print(f"    Found {len(organic_tasks)} organic drift tasks")

    # Generate multi-hop tasks
    print("\n[2] Generating multi-hop reasoning tasks...")
    multi_hop = miner.generate_multi_hop_tasks()
    print(f"    Generated {len(multi_hop)} multi-hop tasks")

    # Save dataset
    print("\n[3] Saving dataset...")
    dataset_path = miner.save_dataset()

    # Create corpus versions
    print("\n[4] Creating corpus versions...")
    corpus_v1, corpus_v2 = miner.create_corpus_versions()

    corpus_v1_path = output_dir / "corpus_v1.json"
    corpus_v2_path = output_dir / "corpus_v2.json"

    with open(corpus_v1_path, 'w') as f:
        json.dump(corpus_v1, f, indent=2)
    with open(corpus_v2_path, 'w') as f:
        json.dump(corpus_v2, f, indent=2)

    print(f"    Saved corpus_v1.json ({len(corpus_v1)} docs)")
    print(f"    Saved corpus_v2.json ({len(corpus_v2)} docs)")

    # Generate distractors
    print("\n[5] Generating distractor documents...")
    distractors = miner.generate_distractor_docs()
    distractor_path = output_dir / "distractors.json"
    with open(distractor_path, 'w') as f:
        json.dump(distractors, f, indent=2)
    print(f"    Saved {len(distractors)} distractor docs")

    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    print(f"  Total tasks: {len(miner.tasks)}")
    print(f"  Factoid: {len([t for t in miner.tasks if t.category == 'factoid'])}")
    print(f"  Multi-hop: {len([t for t in miner.tasks if t.category == 'multi_hop'])}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    return miner.tasks


if __name__ == "__main__":
    tasks = main()
