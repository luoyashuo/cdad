"""
MCC-KD 数据预处理脚本 - DeepMath 版本
实现 Rationale Filtering：基于 N-gram Jaccard 相似度的多样性筛选

输出格式对齐 MCC-KD 官方实现:
{
    "instruction": "问题文本",
    "output_a": "<think>\nthought1\n</think>\n\nsolution1",
    "output_b": "<think>\nthought2\n</think>\n\nsolution2",
    "indices_a": [[start1, end1]],  // answer 在 tokenized output_a 中的位置
    "indices_b": [[start2, end2]]   // answer 在 tokenized output_b 中的位置
}
"""

import json
import random
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import argparse


def extract_ngrams(text: str, n: int = 3) -> Set[str]:
    """
    提取文本的 N-gram 集合
    
    Args:
        text: 输入文本
        n: N-gram 大小，默认为 3
    
    Returns:
        N-gram 集合
    """
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    
    if len(tokens) < n:
        return set(tokens)
    
    ngrams = set()
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i + n])
        ngrams.add(ngram)
    
    return ngrams


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    计算两个集合的 Jaccard 相似度
    
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def extract_answer_length(rationale: Dict[str, str]) -> int:
    """
    提取 rationale 答案部分的长度（\boxed{} 中的内容长度）
    
    Args:
        rationale: 包含 thought 和 solution 的字典
    
    Returns:
        答案文本的长度（字符数），如果没有找到 \boxed{} 返回 -1
    """
    solution = rationale.get("solution", "")
    patterns = [
        r'\\boxed\{([^}]*)\}',  # \boxed{...}
        r'\\boxed\[([^\]]*)\]',  # \boxed[...]
    ]
    
    for pattern in patterns:
        match = re.search(pattern, solution)
        if match:
            return len(match.group(1).strip())
    
    return -1


def rationale_filtering(
    rationales: List[Dict[str, str]], 
    target_count: int = 2, 
    ngram_size: int = 3,
    answer_length_tolerance: int = 20
) -> List[Dict[str, str]]:
    """
    使用基于 N-gram Jaccard 相似度的方法筛选多样化的 rationales
    并确保最终筛选出的 rationales 答案长度一致
    
    算法步骤（与 MCC-KD 官方实现一致）:
    1. 将所有 rationale 转换为 N-gram 集合
    2. 计算所有 rationale 对之间的 Jaccard 相似度
    3. 找出相似度最高的一对，随机保留其中一个
    4. 重复直到保留目标数量的 rationale
    
    新增：步骤 5 - 确保最终结果的答案长度差异在容忍范围内
    """
    if len(rationales) <= target_count:
        return rationales
    
    # 将每个 rationale 转换为 N-gram 集合
    rationale_ngrams = []
    for r in rationales:
        combined_text = r.get("thought", "") + " " + r.get("solution", "")
        ngrams = extract_ngrams(combined_text, ngram_size)
        rationale_ngrams.append(ngrams)
    
    # 工作列表：存储 (原始索引, rationale 数据, N-gram 集合)
    working_list = list(zip(range(len(rationales)), rationales, rationale_ngrams))
    
    # 迭代移除高相似度的 rationale，直到达到目标数量
    while len(working_list) > target_count:
        max_similarity = -1.0
        max_pair_indices = None
        
        # 找出相似度最高的一对
        for i in range(len(working_list)):
            for j in range(i + 1, len(working_list)):
                _, _, ngrams_i = working_list[i]
                _, _, ngrams_j = working_list[j]
                
                sim = jaccard_similarity(ngrams_i, ngrams_j)
                
                if sim > max_similarity:
                    max_similarity = sim
                    max_pair_indices = (i, j)
        
        if max_pair_indices is None:
            break
        
        # 随机保留其中一个，移除另一个
        i, j = max_pair_indices
        keep_idx = random.choice([i, j])
        remove_idx = j if keep_idx == i else i
        
        working_list.pop(remove_idx)
    
    # 返回筛选后的 rationale（保持原始顺序）
    filtered_rationales = [r for _, r, _ in working_list]
    
    # 新增：检查答案长度一致性，如果不符合要求，尝试找到更好的组合
    filtered_rationales = ensure_answer_length_consistency(
        rationales, filtered_rationales, target_count, ngram_size, answer_length_tolerance
    )
    
    return filtered_rationales


def ensure_answer_length_consistency(
    all_rationales: List[Dict[str, str]],
    current_filtered: List[Dict[str, str]],
    target_count: int,
    ngram_size: int,
    tolerance: int
) -> List[Dict[str, str]]:
    """
    确保筛选出的 rationales 答案长度一致
    
    如果当前筛选结果的答案长度差异超过容忍度，尝试从原始数据中
    选择答案长度更相近的组合
    
    Args:
        all_rationales: 原始所有 rationales
        current_filtered: 当前筛选出的 rationales
        target_count: 目标数量
        ngram_size: N-gram 大小
        tolerance: 答案长度差异容忍度
    
    Returns:
        答案长度一致的 rationales 列表
    """
    if len(current_filtered) < 2:
        return current_filtered
    
    # 计算当前结果的答案长度差异
    answer_lengths = [extract_answer_length(r) for r in current_filtered]
    valid_lengths = [l for l in answer_lengths if l >= 0]
    
    if len(valid_lengths) < 2:
        # 如果找不到足够的答案，返回当前结果
        return current_filtered
    
    length_diff = max(valid_lengths) - min(valid_lengths)
    
    # 如果长度差异在容忍范围内，直接返回
    if length_diff <= tolerance:
        return current_filtered
    
    # 尝试找到答案长度更相近的组合
    # 策略：从所有 rationale 中选择答案长度相近且多样性较好的组合
    
    # 首先按答案长度分组
    length_groups = defaultdict(list)
    for i, r in enumerate(all_rationales):
        ans_len = extract_answer_length(r)
        if ans_len >= 0:
            # 将相近长度的分到同一组（按容忍度分组）
            group_key = ans_len // tolerance
            length_groups[group_key].append((i, r))
    
    # 优先从人数最多的组中选择
    sorted_groups = sorted(length_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    for group_key, group_rationales in sorted_groups:
        if len(group_rationales) >= target_count:
            # 在这个组内进行多样性筛选
            group_indices = [idx for idx, _ in group_rationales]
            group_data = [r for _, r in group_rationales]
            
            # 计算组内 N-gram
            group_ngrams = []
            for r in group_data:
                combined_text = r.get("thought", "") + " " + r.get("solution", "")
                ngrams = extract_ngrams(combined_text, ngram_size)
                group_ngrams.append(ngrams)
            
            # 在组内进行筛选
            working_list = list(zip(group_indices, group_data, group_ngrams))
            
            while len(working_list) > target_count:
                max_similarity = -1.0
                max_pair_indices = None
                
                for i in range(len(working_list)):
                    for j in range(i + 1, len(working_list)):
                        _, _, ngrams_i = working_list[i]
                        _, _, ngrams_j = working_list[j]
                        sim = jaccard_similarity(ngrams_i, ngrams_j)
                        if sim > max_similarity:
                            max_similarity = sim
                            max_pair_indices = (i, j)
                
                if max_pair_indices is None:
                    break
                
                i, j = max_pair_indices
                keep_idx = random.choice([i, j])
                remove_idx = j if keep_idx == i else i
                working_list.pop(remove_idx)
            
            result = [r for _, r, _ in working_list]
            
            # 验证答案长度差异
            result_lengths = [extract_answer_length(r) for r in result]
            result_valid_lengths = [l for l in result_lengths if l >= 0]
            
            if len(result_valid_lengths) >= target_count:
                result_diff = max(result_valid_lengths) - min(result_valid_lengths)
                if result_diff <= tolerance:
                    return result
    
    # 如果找不到满足条件的组合，返回原始筛选结果
    return current_filtered


def format_reasoning(thought: str, solution: str) -> str:
    """
    将 thought 和 solution 格式化为指定格式
    
    格式：<think>\n{thought}\n</think>\n\n{solution}
    """
    return f"<think>\n{thought}\n</think>\n\n{solution}"


def extract_boxed_answer_info(solution: str) -> Tuple[str, int, int]:
    """
    从 solution 中提取 \boxed{} 中的答案及其在 solution 中的位置信息
    
    Args:
        solution: 解决方案文本
    
    Returns:
        (answer_text, answer_start_in_solution, answer_end_in_solution)
        如果没有找到 \boxed{}，返回 (None, -1, -1)
    """
    # 匹配 \boxed{...} 或 \boxed[...]
    patterns = [
        r'\\boxed\{([^}]*)\}',  # \boxed{...}
        r'\\boxed\[([^\]]*)\]',  # \boxed[...]
    ]
    
    for pattern in patterns:
        match = re.search(pattern, solution)
        if match:
            answer_text = match.group(1).strip()
            boxed_start = match.start()
            # 找到 { 或 [ 的位置
            brace_start = solution.find('{', boxed_start)
            if brace_start == -1:
                brace_start = solution.find('[', boxed_start)
            if brace_start == -1:
                continue
            answer_start = brace_start + 1
            answer_end = match.end() - 1  # 排除结尾的 } 或 ]
            return answer_text, answer_start, answer_end
    
    return None, -1, -1


def compute_answer_indices(output_text: str, answer_start_in_solution: int, 
                           answer_end_in_solution: int, thought: str) -> List[List[int]]:
    """
    计算 answer 在 formatted output 中的字符位置，并转换为 token 位置索引
    
    注意：这里计算的是字符级别的位置，在训练时需要用 tokenizer 转换为 token 位置
    
    Args:
        output_text: 格式化后的输出文本 (<think>\nthought\n</think>\n\nsolution)
        answer_start_in_solution: answer 在 solution 中的起始位置
        answer_end_in_solution: answer 在 solution 中的结束位置
        thought: thought 文本（用于计算 offset）
    
    Returns:
        [[start_idx, end_idx]] - 与 MCC-KD 格式对齐
    """
    if answer_start_in_solution < 0 or answer_end_in_solution < 0:
        # 没有找到 answer，返回空列表或整个 output 的范围
        return [[0, len(output_text)]]
    
    # 计算 solution 在 output_text 中的起始偏移量
    solution_offset = len(f"<think>\n{thought}\n</think>\n\n")
    
    # 计算 answer 在 output_text 中的绝对字符位置
    answer_start_in_output = solution_offset + answer_start_in_solution
    answer_end_in_output = solution_offset + answer_end_in_solution
    
    return [[answer_start_in_output, answer_end_in_output]]


def process_dataset(
    input_path: str,
    output_path: str,
    sample_size: int = 15000,
    rationales_per_question: int = 2,
    seed: int = 42
):
    """
    处理 deepmath_sft_30k 数据集，应用 MCC-KD 的 Rationale Filtering
    
    输出格式与 MCC-KD 官方实现对齐：
    - instruction: 问题文本
    - output_a: 第一个 rationale 的格式化输出
    - output_b: 第二个 rationale 的格式化输出  
    - indices_a: answer 在 output_a 中的位置 [[start, end]]
    - indices_b: answer 在 output_b 中的位置 [[start, end]]
    
    Args:
        input_path: 原始数据集路径
        output_path: 输出数据集路径
        sample_size: 随机采样的 question 数量
        rationales_per_question: 每个问题保留的 rationale 数量（默认 2，与 MCC-KD 一致）
        seed: 随机种子
    """
    random.seed(seed)
    
    print(f"Loading dataset from {input_path}...")
    
    # 读取所有数据
    all_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_data.append(json.loads(line))
    
    print(f"Total samples in original dataset: {len(all_data)}")
    
    # 随机采样指定数量的 question
    if sample_size < len(all_data):
        selected_data = random.sample(all_data, sample_size)
    else:
        selected_data = all_data
    
    print(f"Selected {len(selected_data)} questions for processing")
    
    # 统计信息
    stats = {
        "total_questions": len(selected_data),
        "questions_with_answer": 0,
        "questions_without_answer": 0,
        "total_output_pairs": 0
    }
    
    # 处理每个问题
    processed_data = []
    for idx, item in enumerate(selected_data):
        question = item.get("question", "")
        reasoning_list = item.get("reasoning", [])
        
        if not reasoning_list:
            continue
        
        # 应用 Rationale Filtering，筛选出 2 个多样化的 rationale
        # 同时确保答案长度一致（差异不超过 20 个字符）
        filtered_rationales = rationale_filtering(
            reasoning_list, 
            target_count=rationales_per_question,
            answer_length_tolerance=20
        )
        
        if len(filtered_rationales) < 2:
            # 如果筛选后不足 2 个，跳过该问题
            continue
        
        # 提取两个 rationale 的信息
        rationale_a = filtered_rationales[0]
        rationale_b = filtered_rationales[1]
        
        thought_a = rationale_a.get("thought", "")
        solution_a = rationale_a.get("solution", "")
        thought_b = rationale_b.get("thought", "")
        solution_b = rationale_b.get("solution", "")
        
        # 格式化 output
        output_a = format_reasoning(thought_a, solution_a)
        output_b = format_reasoning(thought_b, solution_b)
        
        # 提取 answer 位置信息
        answer_text_a, answer_start_a, answer_end_a = extract_boxed_answer_info(solution_a)
        answer_text_b, answer_start_b, answer_end_b = extract_boxed_answer_info(solution_b)
        
        # 计算 indices（字符位置，训练时转换为 token 位置）
        indices_a = compute_answer_indices(output_a, answer_start_a, answer_end_a, thought_a)
        indices_b = compute_answer_indices(output_b, answer_start_b, answer_end_b, thought_b)
        
        # 更新统计
        if answer_text_a is not None and answer_text_b is not None:
            stats["questions_with_answer"] += 1
        else:
            stats["questions_without_answer"] += 1
        
        # 构建训练样本 - 与 MCC-KD 格式对齐
        sample = {
            "instruction": question,
            "output_a": output_a,
            "output_b": output_b,
            "indices_a": indices_a,
            "indices_b": indices_b,
            # 保留原始信息用于调试
            "_answer_text_a": answer_text_a if answer_text_a else "",
            "_answer_text_b": answer_text_b if answer_text_b else "",
            "_thought_a": thought_a,
            "_thought_b": thought_b,
            "_solution_a": solution_a,
            "_solution_b": solution_b
        }
        processed_data.append(sample)
        stats["total_output_pairs"] += 1
        
        # 进度显示
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(selected_data)} questions...")
    
    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in processed_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nProcessed dataset saved to {output_path}")
    
    # 打印统计信息
    print("\nDataset Statistics:")
    print(f"  Original questions: {stats['total_questions']}")
    print(f"  Questions with \\boxed{{}} answer: {stats['questions_with_answer']}")
    print(f"  Questions without \\boxed{{}} answer: {stats['questions_without_answer']}")
    print(f"  Total output pairs: {stats['total_output_pairs']}")
    print(f"  Rationales per question: {rationales_per_question}")
    
    # 新增：统计答案长度一致性
    if processed_data:
        consistent_count = 0
        tolerance = 5
        for sample in processed_data:
            ans_a = sample.get("_answer_text_a", "")
            ans_b = sample.get("_answer_text_b", "")
            if ans_a and ans_b:
                if abs(len(ans_a) - len(ans_b)) <= tolerance:
                    consistent_count += 1
        print(f"  Answer length consistent pairs (±{tolerance} chars): {consistent_count}/{len(processed_data)} ({100*consistent_count/len(processed_data):.1f}%)")
    
    # 打印样例
    if processed_data:
        print("\nSample output (first item):")
        sample = processed_data[0]
        print(f"  Instruction: {sample['instruction'][:100]}...")
        print(f"  Output A length: {len(sample['output_a'])}")
        print(f"  Output B length: {len(sample['output_b'])}")
        print(f"  Indices A: {sample['indices_a']}")
        print(f"  Indices B: {sample['indices_b']}")
        print(f"  Answer A length: {len(sample['_answer_text_a'])}, Answer B length: {len(sample['_answer_text_b'])}")
        print(f"  Answer length diff: {abs(len(sample['_answer_text_a']) - len(sample['_answer_text_b']))}")


def main():
    parser = argparse.ArgumentParser(description="MCC-KD Data Preprocessing for DeepMath")
    parser.add_argument(
        "--input_path",
        type=str,
        default="/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train.jsonl",
        help="Path to the original deepmath dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_mcckd.jsonl",
        help="Path to save the processed dataset"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=15000,
        help="Number of questions to sample"
    )
    parser.add_argument(
        "--rationales_per_question",
        type=int,
        default=2,
        help="Number of rationales to keep per question (fixed to 2 for MCC-KD)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    process_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        rationales_per_question=args.rationales_per_question,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
