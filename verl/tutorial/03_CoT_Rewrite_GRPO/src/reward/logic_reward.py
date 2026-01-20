"""
逻辑奖励 (Logic Reward) 计算模块

R_logic = 1 if FSF(Rewrite_CoT) < FSF(Original_CoT) else 0

FSF (Failed Step Fraction): 推理过程中失败分支的比例
"""

import asyncio
import json
import re
from typing import Optional
import httpx


class FSFAnalyzer:
    """
    FSF (Failed Step Fraction) 分析器

    使用 LLM-as-a-Judge 分析推理过程中的失败分支占比
    """

    ANALYSIS_PROMPT = """你是一个推理过程分析专家。请分析以下推理过程中的"失败分支"占比。

**失败分支定义**: 在推理过程中，尝试了某种方法但最终发现不可行而放弃的分支。

**请按照以下步骤分析**:
1. 识别推理过程中的所有步骤
2. 标记哪些步骤是"失败分支"(尝试后放弃)
3. 计算 FSF = 失败分支数 / 总步骤数

**推理过程**:
{reasoning}

**输出格式** (JSON):
```json
{{
  "total_steps": <总步骤数>,
  "failed_steps": <失败分支数>,
  "fsf": <失败分支占比, 0.0-1.0>
}}
```

请只输出 JSON，不要有其他文字。"""

    def __init__(
        self,
        judge_type: str = "api",
        api_url: str = "http://localhost:8000/v1/chat/completions",
        api_key: str = "EMPTY",
        model_name: str = "gpt-4",
        timeout: float = 30.0,
        max_concurrent: int = 16,
    ):
        """
        初始化 FSF 分析器

        Args:
            judge_type: "api" (外部 API) 或 "local" (本地模型)
            api_url: API 地址
            api_key: API 密钥
            model_name: 模型名称
            timeout: 超时时间 (秒)
            max_concurrent: 最大并发数
        """
        self.judge_type = judge_type
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def _parse_fsf_response(self, response_text: str) -> float:
        """解析 FSF 响应"""
        try:
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return float(data.get("fsf", 0.5))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return 0.5

    async def _analyze_single_async(self, reasoning: str) -> float:
        """异步分析单个推理过程的 FSF"""
        async with self.semaphore:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.api_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model_name,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": self.ANALYSIS_PROMPT.format(reasoning=reasoning)
                                }
                            ],
                            "temperature": 0.0,
                            "max_tokens": 512,
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    response_text = result["choices"][0]["message"]["content"]
                    return self._parse_fsf_response(response_text)
            except Exception as e:
                print(f"[FSFAnalyzer] API 调用失败: {e}")
                return 0.5

    def analyze_single(self, reasoning: str) -> float:
        """同步分析单个推理过程的 FSF"""
        return asyncio.run(self._analyze_single_async(reasoning))

    async def analyze_batch_async(self, reasonings: list[str]) -> list[float]:
        """异步批量分析 FSF"""
        tasks = [self._analyze_single_async(r) for r in reasonings]
        return await asyncio.gather(*tasks)

    def analyze_batch(self, reasonings: list[str]) -> list[float]:
        """同步批量分析 FSF"""
        return asyncio.run(self.analyze_batch_async(reasonings))


def compute_logic_reward(
    original_cots: list[str],
    rewrite_cots: list[str],
    fsf_analyzer: FSFAnalyzer,
) -> list[float]:
    """
    计算逻辑奖励

    R_logic = 1 if FSF(Rewrite) < FSF(Original) else 0

    Args:
        original_cots: 原始 CoT 列表
        rewrite_cots: 改写后 CoT 列表
        fsf_analyzer: FSF 分析器

    Returns:
        rewards: 逻辑奖励列表 (0 或 1)
    """
    original_fsfs = fsf_analyzer.analyze_batch(original_cots)
    rewrite_fsfs = fsf_analyzer.analyze_batch(rewrite_cots)

    rewards = []
    for orig_fsf, rewrite_fsf in zip(original_fsfs, rewrite_fsfs):
        if rewrite_fsf < orig_fsf:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


# 全局单例
_fsf_analyzer: Optional[FSFAnalyzer] = None


def get_fsf_analyzer(**kwargs) -> FSFAnalyzer:
    """获取或创建 FSFAnalyzer 单例"""
    global _fsf_analyzer
    if _fsf_analyzer is None:
        _fsf_analyzer = FSFAnalyzer(**kwargs)
    return _fsf_analyzer
