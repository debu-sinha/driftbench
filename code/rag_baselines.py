"""
DRIFTBENCH RAG Baselines - Vanilla RAG and Oracle-Doc implementations

Implements the core baselines for DRIFTBENCH evaluation.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Try to import optional dependencies
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai not installed. Using mock responses.")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Using simple retrieval.")


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    doc_id: str
    content: str
    score: float


@dataclass
class RAGResponse:
    """Response from a RAG system."""
    answer: str
    confidence: float
    retrieved_docs: List[RetrievalResult]
    expressed_uncertainty: bool
    raw_response: str


class BaseRetriever(ABC):
    """Base class for retrievers."""

    @abstractmethod
    def index(self, corpus: Dict[str, str]):
        """Index a corpus."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant documents."""
        pass


class SimpleRetriever(BaseRetriever):
    """Simple BM25-style retriever using term overlap."""

    def __init__(self):
        self.corpus = {}
        self.doc_tokens = {}

    def index(self, corpus: Dict[str, str]):
        """Index corpus using simple tokenization."""
        self.corpus = corpus
        self.doc_tokens = {}

        for doc_id, content in corpus.items():
            tokens = set(content.lower().split())
            self.doc_tokens[doc_id] = tokens

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve using term overlap scoring."""
        query_tokens = set(query.lower().split())

        scores = []
        for doc_id, doc_tokens in self.doc_tokens.items():
            overlap = len(query_tokens & doc_tokens)
            score = overlap / (len(query_tokens) + 1e-6)
            scores.append((doc_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scores[:top_k]:
            results.append(RetrievalResult(
                doc_id=doc_id,
                content=self.corpus[doc_id],
                score=score
            ))

        return results


class EmbeddingRetriever(BaseRetriever):
    """Embedding-based retriever using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers required for EmbeddingRetriever")
        self.model = SentenceTransformer(model_name)
        self.corpus = {}
        self.embeddings = None
        self.doc_ids = []

    def index(self, corpus: Dict[str, str]):
        """Index corpus using embeddings."""
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        texts = [corpus[doc_id] for doc_id in self.doc_ids]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve using cosine similarity."""
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]

        # Compute similarities
        similarities = np.dot(self.embeddings, query_emb)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            results.append(RetrievalResult(
                doc_id=doc_id,
                content=self.corpus[doc_id],
                score=float(similarities[idx])
            ))

        return results


class BaseGenerator(ABC):
    """Base class for generators."""

    @abstractmethod
    def generate(self, question: str, context: str) -> Tuple[str, float, bool]:
        """Generate answer with confidence and uncertainty flag."""
        pass


class MockGenerator(BaseGenerator):
    """Mock generator for testing without API calls."""

    def generate(self, question: str, context: str) -> Tuple[str, float, bool]:
        """Generate mock response based on context."""
        # Simple extraction: look for patterns in context
        answer = "Unable to determine from context"
        confidence = 0.5
        uncertainty = True

        # Look for common answer patterns
        if "Value:" in context:
            # Extract value after "Value:"
            parts = context.split("Value:")
            if len(parts) > 1:
                answer = parts[1].strip().split()[0].strip(".,;")
                confidence = 0.85
                uncertainty = False
        elif "default" in question.lower():
            # Look for default values
            if "True" in context:
                answer = "True"
                confidence = 0.8
                uncertainty = False
            elif "False" in context:
                answer = "False"
                confidence = 0.8
                uncertainty = False

        return answer, confidence, uncertainty


class OpenAIGenerator(BaseGenerator):
    """OpenAI-based generator."""

    def __init__(self, model: str = "gpt-4o-mini"):
        if not HAS_OPENAI:
            raise ImportError("openai required for OpenAIGenerator")
        self.client = openai.OpenAI()
        self.model = model

    def generate(self, question: str, context: str) -> Tuple[str, float, bool]:
        """Generate answer using OpenAI."""
        prompt = f"""Based on the following documentation, answer the question.

Documentation:
{context}

Question: {question}

Provide your answer in the following JSON format:
{{"answer": "your answer here", "confidence": 0.0-1.0, "uncertain": true/false}}

If you're unsure or the documentation might be outdated, set uncertain to true.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500
            )

            raw = response.choices[0].message.content
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return (
                    result.get("answer", ""),
                    float(result.get("confidence", 0.5)),
                    bool(result.get("uncertain", True))
                )
            else:
                return raw.strip(), 0.5, True

        except Exception as e:
            print(f"OpenAI error: {e}")
            return f"Error: {e}", 0.0, True


class VanillaRAG:
    """Vanilla RAG baseline: retrieve top-k, concatenate, generate."""

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        generator: Optional[BaseGenerator] = None,
        top_k: int = 3
    ):
        self.retriever = retriever or SimpleRetriever()
        self.generator = generator or MockGenerator()
        self.top_k = top_k
        self.corpus = {}

    def index(self, corpus: Dict[str, str]):
        """Index the corpus."""
        self.corpus = corpus
        self.retriever.index(corpus)

    def answer(self, question: str) -> RAGResponse:
        """Answer a question using RAG."""
        # Retrieve
        retrieved = self.retriever.retrieve(question, self.top_k)

        # Build context
        context = "\n\n---\n\n".join([
            f"[Doc {r.doc_id}]\n{r.content}"
            for r in retrieved
        ])

        # Generate
        answer, confidence, uncertainty = self.generator.generate(question, context)

        return RAGResponse(
            answer=answer,
            confidence=confidence,
            retrieved_docs=retrieved,
            expressed_uncertainty=uncertainty,
            raw_response=f"Context: {context[:200]}... Answer: {answer}"
        )


class OracleDoc:
    """Oracle-Doc baseline: inject gold evidence directly."""

    def __init__(self, generator: Optional[BaseGenerator] = None):
        self.generator = generator or MockGenerator()

    def answer(self, question: str, gold_evidence: str) -> RAGResponse:
        """Answer using gold evidence (bypasses retrieval)."""
        # Generate with gold evidence
        answer, confidence, uncertainty = self.generator.generate(question, gold_evidence)

        return RAGResponse(
            answer=answer,
            confidence=confidence,
            retrieved_docs=[RetrievalResult(
                doc_id="gold",
                content=gold_evidence,
                score=1.0
            )],
            expressed_uncertainty=uncertainty,
            raw_response=f"Gold evidence used. Answer: {answer}"
        )


class IterativeRAG:
    """Iterative RAG: multiple retrieval rounds with query reformulation."""

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        generator: Optional[BaseGenerator] = None,
        top_k: int = 3,
        max_rounds: int = 2
    ):
        self.retriever = retriever or SimpleRetriever()
        self.generator = generator or MockGenerator()
        self.top_k = top_k
        self.max_rounds = max_rounds
        self.corpus = {}

    def index(self, corpus: Dict[str, str]):
        """Index the corpus."""
        self.corpus = corpus
        self.retriever.index(corpus)

    def answer(self, question: str) -> RAGResponse:
        """Answer with iterative retrieval."""
        all_retrieved = []
        current_query = question

        for round_num in range(self.max_rounds):
            # Retrieve
            retrieved = self.retriever.retrieve(current_query, self.top_k)
            all_retrieved.extend(retrieved)

            # Build context
            context = "\n\n---\n\n".join([
                f"[Doc {r.doc_id}]\n{r.content}"
                for r in all_retrieved
            ])

            # Try to generate
            answer, confidence, uncertainty = self.generator.generate(question, context)

            # If confident enough, stop
            if confidence > 0.8 and not uncertainty:
                break

            # Otherwise, reformulate query (simple: add key terms from retrieved docs)
            if round_num < self.max_rounds - 1:
                # Extract potential query expansion terms
                for r in retrieved[:1]:
                    words = r.content.split()[:5]
                    current_query = f"{question} {' '.join(words)}"

        return RAGResponse(
            answer=answer,
            confidence=confidence,
            retrieved_docs=all_retrieved,
            expressed_uncertainty=uncertainty,
            raw_response=f"Rounds: {round_num + 1}. Answer: {answer}"
        )


def load_corpus(path: Path) -> Dict[str, str]:
    """Load corpus from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def main():
    """Test RAG baselines."""
    print("=" * 60)
    print("  DRIFTBENCH RAG Baselines Test")
    print("=" * 60)

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    corpus_v1 = load_corpus(data_dir / "corpus_v1.json")
    corpus_v2 = load_corpus(data_dir / "corpus_v2.json")

    with open(data_dir / "fastapi_drift_tasks.json", 'r') as f:
        dataset = json.load(f)

    tasks = dataset["tasks"]
    print(f"\nLoaded {len(tasks)} tasks")

    # Test Vanilla RAG
    print("\n[1] Testing Vanilla RAG on v1 corpus...")
    rag_v1 = VanillaRAG(top_k=3)
    rag_v1.index(corpus_v1)

    correct_v1 = 0
    for task in tasks[:5]:  # Test first 5
        response = rag_v1.answer(task["question"])
        expected = task["answer_v1"]
        is_correct = expected.lower() in response.answer.lower()
        correct_v1 += int(is_correct)
        print(f"  Q: {task['question'][:50]}...")
        print(f"  A: {response.answer[:50]}... (conf: {response.confidence:.2f})")
        print(f"  Expected: {expected} -> {'OK' if is_correct else 'WRONG'}")
        print()

    print(f"  V1 Accuracy: {correct_v1}/5 = {correct_v1/5*100:.0f}%")

    # Test on v2 corpus
    print("\n[2] Testing Vanilla RAG on v2 corpus...")
    rag_v2 = VanillaRAG(top_k=3)
    rag_v2.index(corpus_v2)

    correct_v2 = 0
    for task in tasks[:5]:
        response = rag_v2.answer(task["question"])
        expected = task["answer_v2"]
        is_correct = expected.lower() in response.answer.lower()
        correct_v2 += int(is_correct)

    print(f"  V2 Accuracy: {correct_v2}/5 = {correct_v2/5*100:.0f}%")

    # Test Oracle-Doc
    print("\n[3] Testing Oracle-Doc...")
    oracle = OracleDoc()

    oracle_correct = 0
    for task in tasks[:5]:
        response = oracle.answer(task["question"], task["evidence_v2"])
        expected = task["answer_v2"]
        is_correct = expected.lower() in response.answer.lower()
        oracle_correct += int(is_correct)

    print(f"  Oracle Accuracy: {oracle_correct}/5 = {oracle_correct/5*100:.0f}%")

    print("\n" + "=" * 60)
    print("  Baseline test complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
