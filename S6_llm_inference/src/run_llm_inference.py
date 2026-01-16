#!/usr/bin/env python3
"""Run PhenoGemini-LLM inference on prepared prompts.

Two execution modes are supported:
  * Mock mode (`--mock`) generates deterministic pseudo logits for smoke tests.
  * Real mode loads a Hugging Face causal language model checkpoint and extracts
    the final-token probabilities over gene tokens supplied via `--gene-vocab`.

The script expects a TSV input with columns `query_id` and `prompt`.
Output is a TSV containing (query_id, entrez_gene_id, gene_symbol, gene_token,
probability, rank).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover - optional dependency
    HAVE_TRANSFORMERS = False

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = PACKAGE_ROOT / "data" / "samples" / "sample_prompts.tsv"
DEFAULT_GENE_VOCAB = PACKAGE_ROOT / "data" / "samples" / "sample_gene_vocab.tsv"


@dataclass
class GeneEntry:
    entrez: str
    symbol: str
    token: str
    token_id: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS,
                        help=f"TSV with query_id,prompt (default: {DEFAULT_PROMPTS})")
    parser.add_argument("--gene-vocab", type=Path, default=DEFAULT_GENE_VOCAB,
                        help=f"TSV with gene vocabulary (default: {DEFAULT_GENE_VOCAB})")
    parser.add_argument("--out", type=Path, required=True,
                        help="Destination TSV for gene probabilities")
    parser.add_argument("--model", default=None,
                        help="(Optional) Hugging Face checkpoint for real inference")
    parser.add_argument("--device", default="auto",
                        help="Device for model inference (auto/cpu/cuda)")
    parser.add_argument("--mock", action="store_true",
                        help="Generate deterministic pseudo logits instead of loading a model")
    parser.add_argument("--seed", type=int, default=13,
                        help="Random seed for mock mode")
    parser.add_argument("--max-genes", type=int, default=0,
                        help="Optional cap on genes scored per query (0 means all)")
    parser.add_argument("--chat-template", action="store_true",
                        help="Treat prompt column as chat messages in JSON and apply tokenizer chat template")
    return parser.parse_args()


def load_prompts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    if 'query_id' not in df.columns or 'prompt' not in df.columns:
        raise SystemExit(f"Prompts file {path} must contain query_id and prompt columns")
    return df


def load_gene_vocab(path: Path, max_genes: int = 0) -> List[GeneEntry]:
    df = pd.read_csv(path, sep='\t')
    required = {'entrez_gene_id', 'gene_symbol', 'gene_token'}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Gene vocab {path} missing columns: {', '.join(sorted(missing))}")
    if max_genes and max_genes > 0:
        df = df.head(max_genes)
    entries = [
        GeneEntry(
            entrez=str(row.entrez_gene_id).strip(),
            symbol=str(row.gene_symbol).strip(),
            token=str(row.gene_token).strip(),
        )
        for row in df.itertuples(index=False)
    ]
    return entries


def softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    shifted = values - values.max()
    exps = np.exp(shifted)
    return exps / np.maximum(exps.sum(), 1e-12)


def generate_mock_logits(entries: List[GeneEntry], query_id: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(abs(hash((query_id, seed))) % (2**32))
    values = rng.standard_normal(len(entries))
    return values.astype(np.float32)


def prepare_tokenizer_and_model(model_name: str, device: str):
    if not HAVE_TRANSFORMERS:
        raise SystemExit("transformers is required for real inference; install it or use --mock")
    kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if device == "auto" else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()
    return tokenizer, model


def infer_logits(model, tokenizer, prompt: str, entries: List[GeneEntry], use_chat_template: bool) -> np.ndarray:
    if use_chat_template:
        try:
            messages = json.loads(prompt)
        except json.JSONDecodeError as exc:
            raise ValueError("Prompt must be valid JSON when --chat-template is set") from exc
        encoded = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True)
    else:
        encoded = tokenizer(prompt, return_tensors='pt')
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    logits = outputs.logits[0, -1, :].to('cpu', dtype=torch.float32)
    scores = []
    for entry in entries:
        if entry.token_id is None:
            token_ids = tokenizer(entry.token, add_special_tokens=False)["input_ids"]
            entry.token_id = token_ids[0] if token_ids else None
        if entry.token_id is None or entry.token_id >= logits.numel():
            scores.append(float('-inf'))
        else:
            scores.append(float(logits[entry.token_id]))
    return np.asarray(scores, dtype=np.float32)


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    gene_entries = load_gene_vocab(args.gene_vocab, args.max_genes)

    if args.mock:
        tokenizer = None
        model = None
    else:
        if not args.model:
            raise SystemExit("--model is required unless --mock is specified")
        tokenizer, model = prepare_tokenizer_and_model(args.model, args.device)

    rows: List[dict] = []
    for row in tqdm(prompts.itertuples(index=False), total=len(prompts), desc="llm-infer", unit="query"):
        if args.mock:
            logits = generate_mock_logits(gene_entries, str(row.query_id), args.seed)
        else:
            logits = infer_logits(model, tokenizer, row.prompt, gene_entries, args.chat_template)
        probs = softmax(logits)
        order = np.argsort(-probs)
        for rank, idx in enumerate(order, start=1):
            entry = gene_entries[idx]
            rows.append(
                {
                    "query_id": row.query_id,
                    "entrez_gene_id": entry.entrez,
                    "gene_symbol": entry.symbol,
                    "gene_token": entry.token,
                    "probability": float(probs[idx]),
                    "rank": rank,
                }
            )

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, sep='\t', index=False)
    print(f"[info] wrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
