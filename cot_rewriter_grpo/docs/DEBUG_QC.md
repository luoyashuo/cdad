# 任务
训练任务: bash scripts/grpo/start_qc.sh
运行环境(uv)：
```
source /data/luoys/projects/cdad/cot_rewriter_grpo/.venv/bin/activate
```
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
3. 运行配置是8*A100-80G
4. 你不能修改数据的长度， response长度
# LOG
/data/luoys/projects/cdad/cot_rewriter_grpo/logs/grpo/llama3_8b_grpo_8x80g.log
# 问题
我希望给rewrite_cot的R_cog增加一个新的penalty，repeat_penalty:
如果通过以下方式计算的original_cot和write_cot之间的相似度大于0.9,则给一个固定值的reward惩罚，通过similarity_penalty来控制惩罚的强度，类似format_penalty的实现方式
  from difflib import SequenceMatcher

  def detailed_similarity(a, b):
      sm = SequenceMatcher(None, a, b)
      blocks = sm.get_matching_blocks()  # 获取所有匹配块

      total_match = sum(block.size for block in blocks)

      print("匹配块详情：")
      for i, block in enumerate(blocks[:-1]):  # 最后一个是 (0,0,0) 哨兵
          print(f"  块{i+1}: A[{block.a}:{block.a+block.size}] == B[{block.b}:{block.b+block.size}]")
          print(f"         内容: {repr(a[block.a:block.a+block.size][:50])}")

      similarity = 2 * total_match / (len(a) + len(b))
      print(f"\n总匹配字符: {total_match}")
      print(f"公式: 2×{total_match} / ({len(a)}+{len(b)}) = {similarity:.6f}")

      return similarity