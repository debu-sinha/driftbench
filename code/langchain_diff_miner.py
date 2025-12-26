"""
LangChain Diff Miner - Extract QA pairs from LangChain breaking changes

LangChain has had many breaking changes between versions, especially:
- v0.0.x to v0.1.x (major restructuring)
- v0.1.x to v0.2.x (package splits)
- langchain-core, langchain-community splits
"""

import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class LangChainChange:
    """A single change between LangChain versions."""
    version_old: str
    version_new: str
    change_type: str
    topic: str
    old_value: str
    new_value: str
    context: str
    question: str
    difficulty: str


# LangChain breaking changes (curated from changelogs and migration guides)
LANGCHAIN_BREAKING_CHANGES = [
    # Package structure changes (v0.0.x -> v0.1.x)
    {
        "version_old": "0.0.350",
        "version_new": "0.1.0",
        "changes": [
            {
                "change_type": "import_changed",
                "topic": "ChatOpenAI import",
                "old_value": "from langchain.chat_models import ChatOpenAI",
                "new_value": "from langchain_openai import ChatOpenAI",
                "context": "LangChain split into langchain-core and provider packages",
                "question": "How do you import ChatOpenAI in LangChain v0.1+?",
                "difficulty": "easy"
            },
            {
                "change_type": "import_changed",
                "topic": "OpenAI embeddings import",
                "old_value": "from langchain.embeddings import OpenAIEmbeddings",
                "new_value": "from langchain_openai import OpenAIEmbeddings",
                "context": "Embeddings moved to provider-specific packages",
                "question": "How do you import OpenAI embeddings in LangChain v0.1+?",
                "difficulty": "easy"
            },
            {
                "change_type": "import_changed",
                "topic": "Chroma import",
                "old_value": "from langchain.vectorstores import Chroma",
                "new_value": "from langchain_chroma import Chroma",
                "context": "Vector stores moved to community or dedicated packages",
                "question": "How do you import Chroma vector store in LangChain v0.1+?",
                "difficulty": "medium"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Chain invocation",
                "old_value": "chain.run(input)",
                "new_value": "chain.invoke(input)",
                "context": "LangChain Expression Language (LCEL) uses invoke/batch/stream",
                "question": "What method do you use to run a chain in LangChain v0.1+?",
                "difficulty": "easy"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Async chain execution",
                "old_value": "await chain.arun(input)",
                "new_value": "await chain.ainvoke(input)",
                "context": "Async methods renamed to ainvoke/abatch/astream",
                "question": "What async method runs a chain in LangChain v0.1+?",
                "difficulty": "easy"
            },
        ]
    },
    # LCEL changes
    {
        "version_old": "0.0.300",
        "version_new": "0.1.0",
        "changes": [
            {
                "change_type": "behavior_changed",
                "topic": "Chain composition",
                "old_value": "SequentialChain([chain1, chain2])",
                "new_value": "chain1 | chain2 (pipe operator)",
                "context": "LCEL uses pipe operator for chain composition",
                "question": "How do you compose chains together in LCEL?",
                "difficulty": "medium"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Prompt templates",
                "old_value": "LLMChain(llm=llm, prompt=prompt)",
                "new_value": "prompt | llm (pipe composition)",
                "context": "LLMChain deprecated in favor of LCEL",
                "question": "What replaces LLMChain in LangChain v0.1+?",
                "difficulty": "medium"
            },
            {
                "change_type": "param_renamed",
                "topic": "Output parsing",
                "old_value": "LLMChain with output_key",
                "new_value": "chain | parser (RunnablePassthrough)",
                "context": "Output parsing integrated into LCEL",
                "question": "How do you parse LLM output in LCEL?",
                "difficulty": "hard"
            },
        ]
    },
    # v0.1.x to v0.2.x changes
    {
        "version_old": "0.1.0",
        "version_new": "0.2.0",
        "changes": [
            {
                "change_type": "default_changed",
                "topic": "Callback handling",
                "old_value": "Callbacks passed at chain creation",
                "new_value": "Callbacks passed at invoke time (config={'callbacks': [...]})",
                "context": "LangChain v0.2 prefers config dict for callbacks",
                "question": "How do you pass callbacks to a chain in LangChain v0.2?",
                "difficulty": "medium"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Memory in chains",
                "old_value": "ConversationChain with memory parameter",
                "new_value": "RunnableWithMessageHistory wrapper",
                "context": "Memory handling refactored for LCEL compatibility",
                "question": "How do you add memory to LCEL chains in LangChain v0.2?",
                "difficulty": "hard"
            },
            {
                "change_type": "import_changed",
                "topic": "Document loaders",
                "old_value": "from langchain.document_loaders import TextLoader",
                "new_value": "from langchain_community.document_loaders import TextLoader",
                "context": "Document loaders moved to langchain-community",
                "question": "How do you import TextLoader in LangChain v0.2?",
                "difficulty": "easy"
            },
            {
                "change_type": "import_changed",
                "topic": "Text splitters",
                "old_value": "from langchain.text_splitter import RecursiveCharacterTextSplitter",
                "new_value": "from langchain_text_splitters import RecursiveCharacterTextSplitter",
                "context": "Text splitters moved to dedicated package",
                "question": "How do you import RecursiveCharacterTextSplitter in LangChain v0.2?",
                "difficulty": "easy"
            },
        ]
    },
    # Agent changes
    {
        "version_old": "0.0.350",
        "version_new": "0.1.0",
        "changes": [
            {
                "change_type": "behavior_changed",
                "topic": "Agent creation",
                "old_value": "initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT)",
                "new_value": "create_react_agent(llm, tools, prompt)",
                "context": "Agent creation refactored with explicit prompt",
                "question": "How do you create a ReAct agent in LangChain v0.1+?",
                "difficulty": "medium"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Agent execution",
                "old_value": "agent.run(query)",
                "new_value": "AgentExecutor(agent, tools).invoke({'input': query})",
                "context": "AgentExecutor required for running agents",
                "question": "How do you execute an agent in LangChain v0.1+?",
                "difficulty": "medium"
            },
            {
                "change_type": "import_changed",
                "topic": "Tool decorator",
                "old_value": "from langchain.agents import tool",
                "new_value": "from langchain_core.tools import tool",
                "context": "Tool decorator moved to langchain-core",
                "question": "How do you import the @tool decorator in LangChain v0.1+?",
                "difficulty": "easy"
            },
        ]
    },
    # Retrieval changes
    {
        "version_old": "0.0.300",
        "version_new": "0.1.0",
        "changes": [
            {
                "change_type": "behavior_changed",
                "topic": "RAG chain creation",
                "old_value": "RetrievalQA.from_chain_type(llm, retriever=retriever)",
                "new_value": "create_retrieval_chain(retriever, combine_docs_chain)",
                "context": "RAG chains refactored for LCEL",
                "question": "How do you create a RAG chain in LangChain v0.1+?",
                "difficulty": "hard"
            },
            {
                "change_type": "param_renamed",
                "topic": "Similarity search",
                "old_value": "vectorstore.similarity_search(query, k=4)",
                "new_value": "vectorstore.similarity_search(query, k=4) # unchanged but returns List[Document]",
                "context": "Return type standardized to List[Document]",
                "question": "What does similarity_search return in LangChain?",
                "difficulty": "easy"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Retriever creation",
                "old_value": "vectorstore.as_retriever(search_kwargs={'k': 4})",
                "new_value": "vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 4})",
                "context": "Explicit search_type parameter recommended",
                "question": "How do you specify search type when creating a retriever?",
                "difficulty": "medium"
            },
        ]
    },
    # Output parser changes
    {
        "version_old": "0.0.250",
        "version_new": "0.1.0",
        "changes": [
            {
                "change_type": "import_changed",
                "topic": "PydanticOutputParser import",
                "old_value": "from langchain.output_parsers import PydanticOutputParser",
                "new_value": "from langchain_core.output_parsers import PydanticOutputParser",
                "context": "Output parsers moved to langchain-core",
                "question": "How do you import PydanticOutputParser in LangChain v0.1+?",
                "difficulty": "easy"
            },
            {
                "change_type": "behavior_changed",
                "topic": "JSON output parsing",
                "old_value": "OutputFixingParser with LLM for fixing",
                "new_value": "JsonOutputParser with automatic schema validation",
                "context": "JSON parsing improved with better error handling",
                "question": "What parser is recommended for JSON output in LangChain v0.1+?",
                "difficulty": "medium"
            },
        ]
    },
    # Prompt template changes
    {
        "version_old": "0.0.200",
        "version_new": "0.1.0",
        "changes": [
            {
                "change_type": "import_changed",
                "topic": "ChatPromptTemplate import",
                "old_value": "from langchain.prompts import ChatPromptTemplate",
                "new_value": "from langchain_core.prompts import ChatPromptTemplate",
                "context": "Prompts moved to langchain-core",
                "question": "How do you import ChatPromptTemplate in LangChain v0.1+?",
                "difficulty": "easy"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Message placeholders",
                "old_value": "MessagesPlaceholder(variable_name='history')",
                "new_value": "MessagesPlaceholder('history') or ('placeholder', '{history}')",
                "context": "Multiple ways to define message placeholders",
                "question": "How do you add a messages placeholder in ChatPromptTemplate?",
                "difficulty": "medium"
            },
        ]
    },
    # Streaming changes
    {
        "version_old": "0.0.300",
        "version_new": "0.1.0",
        "changes": [
            {
                "change_type": "behavior_changed",
                "topic": "Streaming output",
                "old_value": "for chunk in llm.stream(prompt): print(chunk)",
                "new_value": "for chunk in chain.stream(input): print(chunk.content)",
                "context": "Streaming returns AIMessageChunk objects",
                "question": "How do you access streamed content in LangChain v0.1+?",
                "difficulty": "medium"
            },
            {
                "change_type": "behavior_changed",
                "topic": "Async streaming",
                "old_value": "async for chunk in llm.astream(prompt)",
                "new_value": "async for chunk in chain.astream(input)",
                "context": "Async streaming uses astream method",
                "question": "What method is used for async streaming in LangChain?",
                "difficulty": "easy"
            },
        ]
    },
    # LangSmith integration
    {
        "version_old": "0.0.350",
        "version_new": "0.1.0",
        "changes": [
            {
                "change_type": "default_changed",
                "topic": "Tracing",
                "old_value": "LANGCHAIN_TRACING_V2=true environment variable",
                "new_value": "LANGCHAIN_TRACING_V2=true with langsmith package",
                "context": "LangSmith tracing requires separate langsmith package",
                "question": "What package is needed for LangSmith tracing in LangChain v0.1+?",
                "difficulty": "easy"
            },
        ]
    },
    # Hub changes
    {
        "version_old": "0.0.300",
        "version_new": "0.1.0",
        "changes": [
            {
                "change_type": "import_changed",
                "topic": "Hub import",
                "old_value": "from langchain import hub",
                "new_value": "from langchain import hub  # or langchainhub package",
                "context": "Hub functionality may require langchainhub package",
                "question": "How do you pull prompts from LangChain Hub?",
                "difficulty": "medium"
            },
        ]
    },
]


def mine_langchain_changes() -> List[Dict]:
    """Extract all LangChain breaking changes as task dicts."""
    tasks = []
    task_id = 0

    for version_block in LANGCHAIN_BREAKING_CHANGES:
        v_old = version_block["version_old"]
        v_new = version_block["version_new"]

        for change in version_block["changes"]:
            task = {
                "task_id": f"langchain_organic_{task_id:04d}",
                "question": change["question"],
                "answer_v1": change["old_value"],
                "answer_v2": change["new_value"],
                "evidence_v1": f"[LangChain {v_old}] {change['context']} Method: {change['old_value']}",
                "evidence_v2": f"[LangChain {v_new}] {change['context']} Method: {change['new_value']}",
                "category": "factoid",
                "source_change": {
                    "file_path": f"docs/{change['topic'].lower().replace(' ', '_')}.md",
                    "change_type": change["change_type"],
                    "old_value": change["old_value"],
                    "new_value": change["new_value"],
                    "context": change["context"],
                    "version_old": v_old,
                    "version_new": v_new
                },
                "difficulty": change["difficulty"]
            }
            tasks.append(task)
            task_id += 1

    return tasks


def save_langchain_dataset(output_dir: Path):
    """Save LangChain drift tasks."""
    tasks = mine_langchain_changes()

    dataset = {
        "name": "DRIFTBENCH-LangChain",
        "version": "0.1",
        "created": datetime.now().isoformat(),
        "n_tasks": len(tasks),
        "categories": {
            "import_changed": len([t for t in tasks if t["source_change"]["change_type"] == "import_changed"]),
            "behavior_changed": len([t for t in tasks if t["source_change"]["change_type"] == "behavior_changed"]),
            "param_renamed": len([t for t in tasks if t["source_change"]["change_type"] == "param_renamed"]),
            "default_changed": len([t for t in tasks if t["source_change"]["change_type"] == "default_changed"]),
        },
        "tasks": tasks
    }

    # Save LangChain-specific dataset
    output_path = output_dir / "langchain_drift_tasks.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(tasks)} LangChain tasks to {output_path}")

    # Create corpus versions
    corpus_v1 = {}
    corpus_v2 = {}

    for task in tasks:
        doc_id = task["task_id"]
        corpus_v1[doc_id] = f"""
# LangChain Documentation

{task["evidence_v1"]}

Related topic: {task["source_change"]["context"]}
"""
        corpus_v2[doc_id] = f"""
# LangChain Documentation

{task["evidence_v2"]}

Related topic: {task["source_change"]["context"]}
"""

    # Save corpora
    with open(output_dir / "langchain_corpus_v1.json", 'w') as f:
        json.dump(corpus_v1, f, indent=2)
    with open(output_dir / "langchain_corpus_v2.json", 'w') as f:
        json.dump(corpus_v2, f, indent=2)

    print(f"Saved LangChain corpus v1: {len(corpus_v1)} docs")
    print(f"Saved LangChain corpus v2: {len(corpus_v2)} docs")

    return tasks


def main():
    """Mine LangChain changes."""
    print("=" * 60)
    print("  DRIFTBENCH - LangChain Diff Mining")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    tasks = save_langchain_dataset(output_dir)

    print(f"\n  Total LangChain tasks: {len(tasks)}")
    print("\n  By change type:")
    for change_type in set(t["source_change"]["change_type"] for t in tasks):
        count = len([t for t in tasks if t["source_change"]["change_type"] == change_type])
        print(f"    {change_type}: {count}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
