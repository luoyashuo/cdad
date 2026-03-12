#!/usr/bin/env python3
"""
提取Claude项目JSONL文件中的对话流
包括：user、assistant、tool_call、tool_result
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def format_timestamp(ts_str):
    """格式化时间戳"""
    if not ts_str:
        return ""
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return ts_str


def extract_content_from_message(msg):
    """从assistant消息中提取内容"""
    content_parts = []
    
    if "message" not in msg or "content" not in msg["message"]:
        return ""
    
    content = msg["message"]["content"]
    if not isinstance(content, list):
        return str(content) if content else ""
    
    for part in content:
        if not isinstance(part, dict):
            continue
        
        part_type = part.get("type", "")
        
        if part_type == "text":
            text = part.get("text", "")
            if text:
                content_parts.append(f"[text]\n{text}")
        
        elif part_type == "thinking":
            thinking = part.get("thinking", "")
            if thinking:
                content_parts.append(f"[thinking]\n{thinking}")
        
        elif part_type == "tool_use":
            tool_name = part.get("name", "")
            tool_input = part.get("input", {})
            tool_id = part.get("id", "")
            content_parts.append(f"[tool_call] {tool_name}\nInput: {json.dumps(tool_input, ensure_ascii=False, indent=2)}")
    
    return "\n\n".join(content_parts)


def extract_tool_result(msg):
    """提取tool_result内容"""
    if "message" not in msg or "content" not in msg["message"]:
        return ""
    
    content = msg["message"]["content"]
    if not isinstance(content, list):
        return ""
    
    results = []
    for part in content:
        if not isinstance(part, dict):
            continue
        
        if part.get("type") == "tool_result":
            tool_use_id = part.get("tool_use_id", "")
            tool_content = part.get("content", "")
            is_error = part.get("is_error", False)
            
            # 处理tool_result的内容
            if isinstance(tool_content, list) and len(tool_content) > 0:
                # 通常是 [{"type": "text", "text": "..."}]
                texts = []
                for c in tool_content:
                    if isinstance(c, dict) and "text" in c:
                        texts.append(c["text"])
                tool_content = "\n".join(texts)
            
            status = "ERROR" if is_error else "OK"
            results.append(f"[tool_result] {tool_use_id}\nStatus: {status}\nContent:\n{tool_content}")
    
    return "\n\n".join(results)


def extract_user_message(msg):
    """提取用户消息内容"""
    if "message" not in msg:
        return ""
    
    message = msg["message"]
    
    # 简单文本消息
    if isinstance(message, str):
        return message
    
    # 包含content字段
    if "content" in message:
        content = message["content"]
        if isinstance(content, str):
            return content
        
        # 可能是包含tool_result的数组
        if isinstance(content, list):
            # 检查是否有普通文本
            texts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    texts.append(c.get("text", ""))
            if texts:
                return "\n".join(texts)
            
            # 否则可能是tool_result
            return extract_tool_result(msg)
    
    return ""


def parse_conversation(jsonl_path):
    """解析JSONL文件并提取对话流"""
    
    entries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    
    # 过滤掉系统事件和快照
    filtered_entries = []
    for entry in entries:
        entry_type = entry.get("type", "")
        if entry_type in ["file-history-snapshot", "progress"]:
            continue
        if entry_type == "system":
            continue
        filtered_entries.append(entry)
    
    return filtered_entries


def build_conversation_flow(entries):
    """构建对话流"""
    
    rounds = []
    current_round = {"user": None, "assistant": [], "tool_calls": [], "tool_results": []}
    
    for entry in entries:
        entry_type = entry.get("type", "")
        
        if entry_type == "user":
            # 如果有之前的对话，保存它
            if current_round["user"] is not None or current_round["assistant"]:
                rounds.append(current_round)
                current_round = {"user": None, "assistant": [], "tool_calls": [], "tool_results": []}
            
            # 提取用户消息
            user_content = extract_user_message(entry)
            timestamp = format_timestamp(entry.get("timestamp", ""))
            current_round["user"] = {
                "content": user_content,
                "timestamp": timestamp,
                "uuid": entry.get("uuid", "")
            }
        
        elif entry_type == "assistant":
            # 提取assistant消息
            content = extract_content_from_message(entry)
            timestamp = format_timestamp(entry.get("timestamp", ""))
            
            # 检查是否包含tool_call
            if "message" in entry and "content" in entry["message"]:
                msg_content = entry["message"]["content"]
                if isinstance(msg_content, list):
                    for part in msg_content:
                        if isinstance(part, dict) and part.get("type") == "tool_use":
                            current_round["tool_calls"].append({
                                "name": part.get("name", ""),
                                "input": part.get("input", {}),
                                "id": part.get("id", ""),
                                "timestamp": timestamp
                            })
            
            if content:
                current_round["assistant"].append({
                    "content": content,
                    "timestamp": timestamp,
                    "uuid": entry.get("uuid", "")
                })
        
        # 检查tool_result（可能在user消息中）
        if "message" in entry and "content" in entry["message"]:
            msg_content = entry["message"]["content"]
            if isinstance(msg_content, list):
                for part in msg_content:
                    if isinstance(part, dict) and part.get("type") == "tool_result":
                        tool_content = part.get("content", "")
                        if isinstance(tool_content, list):
                            texts = []
                            for c in tool_content:
                                if isinstance(c, dict) and "text" in c:
                                    texts.append(c["text"])
                            tool_content = "\n".join(texts)
                        
                        current_round["tool_results"].append({
                            "tool_use_id": part.get("tool_use_id", ""),
                            "content": tool_content,
                            "is_error": part.get("is_error", False),
                            "timestamp": format_timestamp(entry.get("timestamp", ""))
                        })
    
    # 添加最后一个对话轮
    if current_round["user"] is not None or current_round["assistant"]:
        rounds.append(current_round)
    
    return rounds


def format_round(round_num, round_data):
    """格式化单轮对话"""
    lines = []
    lines.append("=" * 80)
    lines.append(f"【第 {round_num} 轮对话】")
    lines.append("=" * 80)
    lines.append("")
    
    # User消息
    if round_data["user"]:
        user = round_data["user"]
        lines.append("-" * 40)
        lines.append(f"[USER] {user.get('timestamp', '')}")
        lines.append("-" * 40)
        lines.append(user.get("content", ""))
        lines.append("")
    
    # Assistant消息
    for idx, assistant in enumerate(round_data["assistant"]):
        lines.append("-" * 40)
        lines.append(f"[ASSISTANT] {assistant.get('timestamp', '')}")
        lines.append("-" * 40)
        lines.append(assistant.get("content", ""))
        lines.append("")
    
    # Tool Calls
    for tool_call in round_data["tool_calls"]:
        lines.append("-" * 40)
        lines.append(f"[TOOL_CALL] {tool_call.get('name', '')} {tool_call.get('timestamp', '')}")
        lines.append("-" * 40)
        lines.append(f"Tool ID: {tool_call.get('id', '')}")
        lines.append("Input:")
        lines.append(json.dumps(tool_call.get("input", {}), ensure_ascii=False, indent=2))
        lines.append("")
    
    # Tool Results
    for tool_result in round_data["tool_results"]:
        lines.append("-" * 40)
        lines.append(f"[TOOL_RESULT] {tool_result.get('timestamp', '')}")
        lines.append("-" * 40)
        lines.append(f"Tool Use ID: {tool_result.get('tool_use_id', '')}")
        if tool_result.get("is_error"):
            lines.append("Status: ERROR")
        lines.append("Content:")
        content = tool_result.get("content", "")
        # 截断过长的内容
        if len(content) > 5000:
            content = content[:5000] + "\n... [内容过长，已截断]"
        lines.append(content)
        lines.append("")
    
    return "\n".join(lines)


def main():
    # 输入输出路径
    input_path = "/home/luoys/.claude/projects/-home-luoys-cdad-cot-rewriter-grpo/bbee6bd3-ebb7-4819-8baa-dd1d8b951cb6.jsonl"
    output_path = "/home/luoys/cdad/cot_rewriter_grpo/scripts/grpo/b"
    
    print(f"正在解析: {input_path}")
    
    # 解析JSONL
    entries = parse_conversation(input_path)
    print(f"共解析 {len(entries)} 条记录")
    
    # 构建对话流
    rounds = build_conversation_flow(entries)
    print(f"共 {len(rounds)} 轮对话")
    
    # 格式化输出
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("Claude 对话记录提取")
    output_lines.append(f"源文件: {input_path}")
    output_lines.append(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"总轮数: {len(rounds)}")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    for i, round_data in enumerate(rounds, 1):
        output_lines.append(format_round(i, round_data))
    
    output_lines.append("=" * 80)
    output_lines.append("【对话记录结束】")
    output_lines.append("=" * 80)
    
    # 写入输出文件
    output_text = "\n".join(output_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_text)
    
    print(f"已保存到: {output_path}")
    print(f"文件大小: {len(output_text)} 字符")


if __name__ == "__main__":
    main()
