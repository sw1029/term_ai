from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from term_ai.contracts import dumps_jsonl


def cosine(left: list[float], right: list[float]) -> float:
    import math

    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def option_top2_similarity(model: Any, options: list[str]) -> float:
    vectors = model.encode(options, normalize_embeddings=True)
    scores: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            scores.append(float(cosine(vectors[i], vectors[j])))
    return max(scores) if scores else 0.0


def add_embedding_similarity(
    metadata_path: str | Path,
    output_path: str | Path,
    model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
    high_similarity_threshold: float = 0.92,
) -> dict[str, int]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install baseline dependencies first: pip install -e .[baseline]") from exc

    model = SentenceTransformer(model_name)
    counts = {"written": 0, "high_similarity": 0}
    with open(metadata_path, "r", encoding="utf-8") as input_handle, open(
        output_path, "w", encoding="utf-8", newline="\n"
    ) as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            payload = row.get("payload") or {}
            options = [str(option) for option in payload.get("options") or []]
            if len(options) == 4:
                sim = option_top2_similarity(model, options)
                payload["embedding_top2_similarity"] = sim
                if sim >= high_similarity_threshold:
                    counts["high_similarity"] += 1
            row["payload"] = payload
            counts["written"] += 1
            output_handle.write(dumps_jsonl(row))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Add embedding based validation signals.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--high-similarity-threshold", type=float, default=0.92)
    args = parser.parse_args()
    counts = add_embedding_similarity(
        metadata_path=args.metadata,
        output_path=args.output,
        model_name=args.model_name,
        high_similarity_threshold=args.high_similarity_threshold,
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
