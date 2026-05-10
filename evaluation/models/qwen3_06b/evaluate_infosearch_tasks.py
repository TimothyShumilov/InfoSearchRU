"""
Evaluate Qwen/Qwen3-Embedding-0.6B on the InfoSearch benchmark tasks
with the same metrics tracked during bi-encoder training.

Usage:
    python evaluate_infosearch_tasks.py [--model_name_or_path PATH] \
                                        [--output_dir DIR] \
                                        [--batch_size N]
"""

import os
import re
import json
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from typing import List, Dict
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from mteb import MTEB
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel, is_dres_compatible
from utils import pool, move_to_cuda, create_batch_dict, \
    get_detailed_instruct, get_task_def_by_task_name_and_type
from model_config import MODEL_NAME_TO_POOL_TYPE, MODEL_NAME_TO_PREFIX_TYPE

# ── InfoSearch benchmark configuration (mirrors training notebook) ────────────
TASK_NAMES = [
    "Clarity-v1-ru",
    "Source-v1-ru",
    "Audience-v1-ru",
    "Language-v1-ru",
    "Length-v1-ru",
    "InstructedRetrieval-val-ru",
]

METRICS = [
    "SICR",
    "WISE",
    "p-MRR",
    "map_at_1000_instruction",
    "map_at_1000_original",
    "map_at_1000_reversed",
    "ndcg_at_10_instruction",
    "ndcg_at_10_original",
    "ndcg_at_10_reversed",
]

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate Qwen3-Embedding on InfoSearch tasks")
parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-Embedding-0.6B",
                    type=str, help="Model identifier or local path")
parser.add_argument("--output_dir", default="qwen3-embedding-0.6b-infosearch",
                    type=str, help="Directory for MTEB result JSONs")
parser.add_argument("--batch_size", default=32, type=int, help="Encoding batch size")
args = parser.parse_args()

base_name: str = args.model_name_or_path.split("/")[-1]
pool_type   = MODEL_NAME_TO_POOL_TYPE.get(base_name, "cls")
prefix_type = MODEL_NAME_TO_PREFIX_TYPE.get(base_name, "instruction")

os.makedirs(args.output_dir, exist_ok=True)

_GENERIC_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def _split_query_instruction(text: str):
    """Split on the last sentence boundary.

    Returns (query_part, instruction_part) where instruction_part is the
    last sentence and query_part is everything before it.
    """
    sentences = [s for s in re.split(r'(?<=[.?!])\s+', text.strip()) if s]
    if len(sentences) < 2:
        return text, ""
    return " ".join(sentences[:-1]), sentences[-1]


# ── Model (mirrors evaluate_qwen3_06b.py) ────────────────────────────────────
class RetrievalModel(DRESModel):
    def __init__(self, **kwargs):
        self.encoder = AutoModel.from_pretrained(
            args.model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prompt = None
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
        if self.gpu_count > 0:
            self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        instructions = kwargs.get("instructions", {})
        input_texts = []
        for q in queries:
            instr = (instructions.get(q) or "").strip()
            if instr:
                input_texts.append(f"Instruct: {instr}\nQuery: {q}")
            else:
                input_texts.append(q)
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        input_texts = ["{} {}".format(doc.get("title", ""), doc["text"]).strip()
                       for doc in corpus]
        if prefix_type == "query_or_passage":
            input_texts = [f"passage: {t}" for t in input_texts]
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        encoded_embeds = []
        batch_size = 64 * self.gpu_count if self.gpu_count > 0 else 1
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size),
                                   desc="encoding", mininterval=10):
            batch_input_texts = input_texts[start_idx: start_idx + batch_size]
            batch_dict = create_batch_dict(
                self.tokenizer, batch_input_texts, always_add_eos=(pool_type == "last")
            )
            if self.gpu_count > 0:
                batch_dict = move_to_cuda(batch_dict)
                with torch.cuda.amp.autocast():
                    outputs: BaseModelOutput = self.encoder(**batch_dict)
                    embeds = pool(outputs.last_hidden_state, batch_dict["attention_mask"], pool_type)
                    embeds = F.normalize(embeds, p=2, dim=-1)
                    encoded_embeds.append(embeds.cpu().numpy())
            else:
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict["attention_mask"], pool_type)
                embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.numpy())
        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt


# ── Metric extraction (mirrors training notebook) ────────────────────────────
def _parse_metrics_from_json(task_json: dict, metrics: List[str]) -> Dict[str, float]:
    """
    Support both mteb_infosearch format (result["test"][metric])
    and stock MTEB 1.x format (result["scores"]["test"][0][metric]).
    """
    test_section = task_json.get("test")
    if not isinstance(test_section, dict):
        scores = task_json.get("scores", {})
        test_list = scores.get("test", [])
        test_section = test_list[0] if test_list else {}

    result = {}
    for m in metrics:
        val = test_section.get(m)
        result[m] = float(val) if val is not None else float("nan")
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    assert is_dres_compatible(RetrievalModel)
    model = RetrievalModel()

    all_results: Dict[str, Dict[str, float]] = {}

    for task_name in TASK_NAMES:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        task_def = get_task_def_by_task_name_and_type(task_name, "InstructionRetrieval")
        model.set_prompt(get_detailed_instruct(task_def))

        evaluation = MTEB(tasks=[task_name], task_langs=["ru"], do_length_ablation=False)
        evaluation.run(
            model,
            eval_splits=["test"],
            output_folder=args.output_dir,
            save_corpus_embeddings=True,
            batch_size=args.batch_size,
            verbosity=0,
        )

        json_path = os.path.join(args.output_dir, f"{task_name}.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                task_json = json.load(f)
            task_metrics = _parse_metrics_from_json(task_json, METRICS)
        else:
            print(f"WARNING: result file not found at {json_path}")
            task_metrics = {m: float("nan") for m in METRICS}

        all_results[task_name] = task_metrics

    # ── Summary table ─────────────────────────────────────────────────────────
    col_w = 10
    name_w = max(len(t) for t in TASK_NAMES) + 2
    short_metrics = [m.replace("map_at_1000_", "map@1k_")
                      .replace("ndcg_at_10_", "ndcg@10_") for m in METRICS]

    sep = "─" * (name_w + col_w * len(METRICS))
    header = f"{'Task':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in short_metrics)

    print(f"\n\n{'='*len(sep)}")
    print(f"Results: {args.model_name_or_path}")
    print(f"{'='*len(sep)}")
    print(header)
    print(sep)

    averages = {m: [] for m in METRICS}
    for task_name in TASK_NAMES:
        row = f"{task_name:<{name_w}}"
        for m in METRICS:
            v = all_results[task_name][m]
            row += f"{v:>{col_w}.4f}" if not math.isnan(v) else f"{'N/A':>{col_w}}"
            if not math.isnan(v):
                averages[m].append(v)
        print(row)

    print(sep)
    avg_row = f"{'Average':<{name_w}}"
    for m in METRICS:
        vals = averages[m]
        avg = sum(vals) / len(vals) if vals else float("nan")
        avg_row += f"{avg:>{col_w}.4f}" if not math.isnan(avg) else f"{'N/A':>{col_w}}"
    print(avg_row)
    print()


if __name__ == "__main__":
    main()
