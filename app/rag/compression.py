"""Phase 8a: context compression. Chunk a long document, retrieve only
the chunks relevant to the actual query, and send that instead of the
whole thing - directly cuts token cost on long-context requests.

ON "SEMANTIC": this uses TF-IDF (term frequency) + cosine similarity, not
real embeddings. That's lexical similarity - it catches shared
vocabulary, not shared meaning ("car" and "automobile" won't match each
other even though they mean the same thing). Genuine semantic embeddings
(Gemini's embedding API, or a local Ollama embedding model) are a natural
upgrade path here - documented as a known limitation rather than quietly
overselling what this actually does, same honesty rule as every other
phase this session.
"""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_text(text: str, chunk_size_words: int = 200) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + chunk_size_words]) for i in range(0, len(words), chunk_size_words)]


@dataclass
class CompressionResult:
    query: str
    compressed_text: str
    original_word_count: int
    compressed_word_count: int
    reduction_pct: float
    chunks_kept: int
    chunks_total: int
    selected_chunk_indices: list[int]

    @property
    def original_est_tokens(self) -> int:
        # ~0.75 words/token (the common inverse of "~1.3 tokens/word" for
        # English) — a different, independent rough estimate from Phase
        # 3's ~4-chars/token rule, since only word counts are stored here,
        # not the original document's raw character count.
        return max(1, round(self.original_word_count / 0.75))

    @property
    def compressed_est_tokens(self) -> int:
        return max(1, round(self.compressed_word_count / 0.75))


def compress_context(query: str, document: str, top_k: int = 3, chunk_size_words: int = 200) -> CompressionResult:
    chunks = chunk_text(document, chunk_size_words)
    original_word_count = len(document.split())

    if len(chunks) <= top_k:
        # Already short enough - nothing meaningful to compress.
        return CompressionResult(
            query=query,
            compressed_text=document,
            original_word_count=original_word_count,
            compressed_word_count=original_word_count,
            reduction_pct=0.0,
            chunks_kept=len(chunks),
            chunks_total=len(chunks),
            selected_chunk_indices=list(range(len(chunks))),
        )

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([query] + chunks)
        similarities = cosine_similarity(matrix[0:1], matrix[1:])[0]
        ranked_indices = similarities.argsort()[::-1][:top_k]
        selected_indices = sorted(ranked_indices.tolist())  # preserve original reading order
    except ValueError:
        # Degenerate input (e.g. everything is stopwords) - fall back to
        # the first top_k chunks rather than crashing.
        selected_indices = list(range(min(top_k, len(chunks))))

    compressed_chunks = [chunks[i] for i in selected_indices]
    compressed_text = "\n\n".join(compressed_chunks)
    compressed_word_count = len(compressed_text.split())
    reduction_pct = (
        (original_word_count - compressed_word_count) / original_word_count * 100 if original_word_count else 0.0
    )

    return CompressionResult(
        query=query,
        compressed_text=compressed_text,
        original_word_count=original_word_count,
        compressed_word_count=compressed_word_count,
        reduction_pct=reduction_pct,
        chunks_kept=len(selected_indices),
        chunks_total=len(chunks),
        selected_chunk_indices=selected_indices,
    )
