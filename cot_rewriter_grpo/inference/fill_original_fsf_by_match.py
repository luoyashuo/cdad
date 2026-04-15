"""
Fill missing original-* FSF fields in fsf_tag_output.jsonl by matching records.

Primary strategy:
  1. Match by (question, original_cot) against donor fsf_tag_output.jsonl files.
  2. Fallback to original_cot-only matching when it is unique across donor files.
  3. If original_fsf is still missing and original_fsf_model_output exists,
     parse the DOT graph locally.

This avoids re-running the FSF judge for old intermediate files that missed
`original_fsf`, `original_fsf_model_output`, or both.
"""

import argparse
import contextlib
import glob
import io
import json
import os
import re
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


Record = Dict[str, Any]
Key = Tuple[str, str]
DonorPayload = Dict[str, Any]


def _read_jsonl(path: str) -> List[Record]:
    records: List[Record] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _record_key(record: Record) -> Key:
    return (
        (record.get("question") or "").strip(),
        (record.get("original_cot") or "").strip(),
    )


def _original_cot_key(record: Record) -> str:
    return (record.get("original_cot") or "").strip()


def _coerce_original_fsf(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_donor_payload(record: Record) -> DonorPayload:
    payload: DonorPayload = {}

    original_fsf = _coerce_original_fsf(record.get("original_fsf"))
    if original_fsf is not None:
        payload["original_fsf"] = original_fsf

    original_fsf_model_output = record.get("original_fsf_model_output")
    if original_fsf_model_output:
        payload["original_fsf_model_output"] = original_fsf_model_output

    return payload


def _parse_dot_fsf(dot_string: Optional[str]) -> Optional[float]:
    if not dot_string:
        return None

    cleaned = dot_string.lstrip("\n").strip("`").strip()

    # First try a permissive text-level parser. Many bad DOT outputs still preserve
    # one-node-per-line declarations, which is enough for FSF counting.
    node_count = 0
    failed_nodes = 0
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or "->" in line or "[" not in line:
            continue

        node_id = line.split("[", 1)[0].strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", node_id):
            continue

        node_count += 1
        if re.search(r'fillcolor\s*=\s*"?lightpink"?', line, flags=re.IGNORECASE):
            failed_nodes += 1

    if node_count > 0:
        return failed_nodes / node_count

    try:
        import pydot

        with contextlib.redirect_stderr(io.StringIO()):
            graphs = pydot.graph_from_dot_data(cleaned)
        if graphs:
            graph = graphs[0]
            nodes = graph.get_nodes() or []
            node_count = len(nodes)
            if node_count == 0:
                return 0.0

            failed_nodes = 0
            for node in nodes:
                if node is None:
                    continue
                try:
                    fillcolor = node.get("fillcolor")
                except Exception:
                    fillcolor = None
                if fillcolor is not None and str(fillcolor).strip('"').lower() == "lightpink":
                    failed_nodes += 1

            return failed_nodes / node_count
    except Exception:
        pass
    return None


def _iter_donor_paths(patterns: Iterable[str], target_path: str) -> List[str]:
    paths = set()
    target_realpath = os.path.realpath(target_path)
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.realpath(path) == target_realpath:
                continue
            if os.path.isfile(path):
                paths.add(path)
    return sorted(paths)


def _payload_signature(payload: DonorPayload) -> Tuple[Optional[float], Optional[str]]:
    return (
        payload.get("original_fsf"),
        payload.get("original_fsf_model_output"),
    )


def _build_donor_index(
    paths: Iterable[str],
) -> Tuple[Dict[Key, DonorPayload], Dict[str, DonorPayload], Dict[str, int]]:
    exact_index: Dict[Key, DonorPayload] = {}
    cot_payload_candidates: Dict[str, Set[Tuple[Optional[float], Optional[str]]]] = {}
    cot_payload_lookup: Dict[str, DonorPayload] = {}
    stats = {
        "exact_conflicts": 0,
        "cot_conflicts": 0,
    }

    for path in paths:
        for record in _read_jsonl(path):
            payload = _extract_donor_payload(record)
            if not payload:
                continue

            key = _record_key(record)
            prev = exact_index.get(key)
            if prev is None:
                exact_index[key] = payload
            elif _payload_signature(prev) != _payload_signature(payload):
                stats["exact_conflicts"] += 1

            cot_key = _original_cot_key(record)
            if not cot_key:
                continue
            cot_payload_candidates.setdefault(cot_key, set()).add(_payload_signature(payload))
            cot_payload_lookup.setdefault(cot_key, payload)

    cot_index: Dict[str, DonorPayload] = {}
    for cot_key, signatures in cot_payload_candidates.items():
        if len(signatures) == 1:
            cot_index[cot_key] = cot_payload_lookup[cot_key]
        else:
            stats["cot_conflicts"] += 1

    return exact_index, cot_index, stats


def _write_jsonl_atomic(path: str, records: List[Record]) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path) or ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _is_missing_model_output(record: Record) -> bool:
    return not bool(record.get("original_fsf_model_output"))


def fill_missing_original_fsf(target_path: str, donor_patterns: List[str]) -> Dict[str, int]:
    records = _read_jsonl(target_path)
    donor_paths = _iter_donor_paths(donor_patterns, target_path)
    exact_donor_index, cot_donor_index, donor_stats = _build_donor_index(donor_paths)

    total = len(records)
    changed = False
    missing_original_fsf_before = 0
    missing_original_fsf_model_output_before = 0
    filled_original_fsf_by_exact_match = 0
    filled_original_fsf_by_cot_match = 0
    filled_original_fsf_model_output_by_exact_match = 0
    filled_original_fsf_model_output_by_cot_match = 0
    filled_original_fsf_by_dot = 0
    still_missing_original_fsf = 0
    still_missing_original_fsf_model_output = 0

    for record in records:
        needs_original_fsf = record.get("original_fsf") is None
        needs_model_output = _is_missing_model_output(record)
        if not needs_original_fsf and not needs_model_output:
            continue

        if needs_original_fsf:
            missing_original_fsf_before += 1
        if needs_model_output:
            missing_original_fsf_model_output_before += 1

        key = _record_key(record)
        matched_payload = exact_donor_index.get(key)
        match_type = "exact"
        if matched_payload is None:
            matched_payload = cot_donor_index.get(_original_cot_key(record))
            match_type = "cot"

        if matched_payload is not None:
            if needs_model_output and matched_payload.get("original_fsf_model_output"):
                record["original_fsf_model_output"] = matched_payload["original_fsf_model_output"]
                changed = True
                if match_type == "exact":
                    filled_original_fsf_model_output_by_exact_match += 1
                else:
                    filled_original_fsf_model_output_by_cot_match += 1

            if needs_original_fsf and matched_payload.get("original_fsf") is not None:
                record["original_fsf"] = matched_payload["original_fsf"]
                changed = True
                if match_type == "exact":
                    filled_original_fsf_by_exact_match += 1
                else:
                    filled_original_fsf_by_cot_match += 1

        if record.get("original_fsf") is None:
            parsed = _parse_dot_fsf(record.get("original_fsf_model_output"))
            if parsed is not None:
                record["original_fsf"] = parsed
                changed = True
                filled_original_fsf_by_dot += 1

        if record.get("original_fsf") is None:
            still_missing_original_fsf += 1
        if _is_missing_model_output(record):
            still_missing_original_fsf_model_output += 1

    if changed:
        _write_jsonl_atomic(target_path, records)

    return {
        "total": total,
        "donor_files": len(donor_paths),
        "donor_exact_keys": len(exact_donor_index),
        "donor_unique_original_cots": len(cot_donor_index),
        "exact_conflicts": donor_stats["exact_conflicts"],
        "cot_conflicts": donor_stats["cot_conflicts"],
        "missing_original_fsf_before": missing_original_fsf_before,
        "missing_original_fsf_model_output_before": missing_original_fsf_model_output_before,
        "filled_original_fsf_by_exact_match": filled_original_fsf_by_exact_match,
        "filled_original_fsf_by_cot_match": filled_original_fsf_by_cot_match,
        "filled_original_fsf_model_output_by_exact_match": filled_original_fsf_model_output_by_exact_match,
        "filled_original_fsf_model_output_by_cot_match": filled_original_fsf_model_output_by_cot_match,
        "filled_original_fsf_by_dot": filled_original_fsf_by_dot,
        "still_missing_original_fsf": still_missing_original_fsf,
        "still_missing_original_fsf_model_output": still_missing_original_fsf_model_output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing original_fsf/original_fsf_model_output values by matching other fsf_tag_output files."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target fsf_tag_output.jsonl to update in-place.",
    )
    parser.add_argument(
        "--donor-glob",
        action="append",
        required=True,
        help="Glob pattern for donor fsf_tag_output.jsonl files. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = fill_missing_original_fsf(args.target, args.donor_glob)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
