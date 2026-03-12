---
name: claude-session-extractor
description: Extract conversation flow from Claude Code session JSONL files into clean markdown format. Use when user needs to convert Claude Code session exports (.jsonl files from ~/.claude/projects/) into readable dialogue format with USER, ASSISTANT, TOOL_CALL, and TOOL_RESULT sections.
---

# Claude Session Extractor

Extract clean conversation flow from Claude Code session JSONL files.

## Usage

```bash
python3 scripts/extract.py <input.jsonl> [output.md]
```

## Output Format

```markdown
# Round 1

## User
User message text...

<tool_result>
Tool output content...
</tool_result>

## Assistant
Assistant response text...

<tool_call> Read
{
  "file_path": "/path/to/file"
}
</tool_call>
```

## Features

- Extracts USER, ASSISTANT, TOOL_CALL, TOOL_RESULT in sequence
- No content truncation - full tool outputs preserved
- Clean format without timestamps or internal IDs
- Tool calls wrapped in `<tool_call> name ... </tool_call>`
- Tool results wrapped in `<tool_result> ... </tool_result>`
- Thinking blocks preserved as `<thinking> ... </thinking>`

## Session File Location

Claude Code session files are stored in:
- `~/.claude/projects/<project-id>/<session-id>.jsonl`

## Example

```bash
# Extract session to markdown
python3 scripts/extract.py ~/.claude/projects/my-project/session.jsonl conversation.md
```
