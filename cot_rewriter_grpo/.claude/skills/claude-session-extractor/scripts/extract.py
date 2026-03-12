#!/usr/bin/env python3
"""
Extract conversation flow from Claude Code session JSONL file.
Output format: Clean dialogue with USER, ASSISTANT, TOOL_CALL, TOOL_RESULT sections.
"""

import json
import sys
from pathlib import Path


def is_tool_result_only(entry):
    """Check if entry is a tool_result message (not a real user message)."""
    if entry.get("type") != "user":
        return False
    if "message" not in entry or "content" not in entry["message"]:
        return False
    
    content = entry["message"]["content"]
    if not isinstance(content, list):
        return False
    
    # Check if all items are tool_result
    return all(
        isinstance(c, dict) and c.get("type") == "tool_result"
        for c in content
    )


def is_real_user_message(entry):
    """Check if entry is a real user message (not tool_result)."""
    if entry.get("type") != "user":
        return False
    return not is_tool_result_only(entry)


def extract_assistant_content(entry):
    """Extract assistant message content (text and tool_calls)."""
    parts = []
    
    if "message" not in entry or "content" not in entry["message"]:
        return ""
    
    content = entry["message"]["content"]
    if not isinstance(content, list):
        text = str(content) if content else ""
        return text
    
    for part in content:
        if not isinstance(part, dict):
            continue
        
        ptype = part.get("type", "")
        
        if ptype == "text":
            text = part.get("text", "")
            if text:
                parts.append(text)
        
        elif ptype == "thinking":
            thinking = part.get("thinking", "")
            if thinking:
                parts.append(f"<thinking>\n{thinking}\n</thinking>")
        
        elif ptype == "tool_use":
            name = part.get("name", "")
            inp = part.get("input", {})
            parts.append(f"<tool_call> {name}\n{json.dumps(inp, ensure_ascii=False, indent=2)}\n</tool_call>")
    
    return "\n\n".join(parts)


def extract_tool_result_content(entry):
    """Extract tool result content."""
    if "message" not in entry or "content" not in entry["message"]:
        return ""
    
    content = entry["message"]["content"]
    if not isinstance(content, list):
        return ""
    
    results = []
    for part in content:
        if not isinstance(part, dict):
            continue
        
        if part.get("type") == "tool_result":
            tc = part.get("content", "")
            
            # Handle content format
            if isinstance(tc, list):
                texts = []
                for c in tc:
                    if isinstance(c, dict) and "text" in c:
                        texts.append(c["text"])
                tc = "\n".join(texts)
            
            results.append(f"<tool_result>\n{tc}\n</tool_result>")
    
    return "\n\n".join(results)


def extract_user_text(entry):
    """Extract real user message text."""
    if "message" not in entry:
        return ""
    
    msg = entry["message"]
    
    if isinstance(msg, str):
        return msg
    
    if "content" not in msg:
        return ""
    
    content = msg["content"]
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        texts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                texts.append(c.get("text", ""))
        return "\n".join(texts)
    
    return ""


def parse_entries(jsonl_path):
    """Parse JSONL and filter relevant entries."""
    entries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Skip system entries
                if entry.get("type") in ["file-history-snapshot", "progress", "system", "queue-operation", "last-prompt"]:
                    continue
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries


def build_rounds(entries):
    """Group entries into conversation rounds."""
    rounds = []
    current = {"user": None, "items": []}
    
    for entry in entries:
        etype = entry.get("type", "")
        
        if etype == "user":
            if is_real_user_message(entry):
                # Start new round
                if current["user"] is not None or current["items"]:
                    rounds.append(current)
                text = extract_user_text(entry)
                current = {"user": text, "items": []}
            else:
                # Tool result - add to current round
                tr = extract_tool_result_content(entry)
                if tr:
                    current["items"].append(("tool_result", tr))
        
        elif etype == "assistant":
            content = extract_assistant_content(entry)
            if content:
                current["items"].append(("assistant", content))
    
    # Add last round
    if current["user"] is not None or current["items"]:
        rounds.append(current)
    
    return rounds


def format_round(n, r):
    """Format a single round."""
    lines = []
    lines.append(f"# Round {n}")
    lines.append("")
    
    if r["user"]:
        lines.append("## User")
        lines.append("")
        lines.append(r["user"])
        lines.append("")
    
    for item_type, content in r["items"]:
        if item_type == "tool_result":
            lines.append(content)
            lines.append("")
        elif item_type == "assistant":
            lines.append("## Assistant")
            lines.append("")
            lines.append(content)
            lines.append("")
    
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: extract.py <input.jsonl> [output.md]", file=sys.stderr)
        sys.exit(1)
    
    inpath = Path(sys.argv[1])
    outpath = Path(sys.argv[2]) if len(sys.argv) > 2 else inpath.with_suffix('.md')
    
    entries = parse_entries(inpath)
    rounds = build_rounds(entries)
    
    lines = []
    lines.append(f"# Claude Session: {inpath.name}")
    lines.append("")
    lines.append(f"Total rounds: {len(rounds)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    for i, r in enumerate(rounds, 1):
        lines.append(format_round(i, r))
        lines.append("")
        lines.append("---")
        lines.append("")
    
    text = "\n".join(lines)
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Extracted {len(rounds)} rounds to {outpath}")


if __name__ == "__main__":
    main()
