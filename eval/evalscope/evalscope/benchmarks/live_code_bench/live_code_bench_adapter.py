# flake8: noqa: E501
import os
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import convert_normal_types, jsonl_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()

_LCB_LOCAL_PATH = '/fs/fast/u20240075/hf_home/datasets/livecodebench/code_generation_lite'

# 版本 -> 文件列表的映射（与 code_generation_lite.py 中的 ALLOWED_FILES 保持一致）
_LCB_VERSION_FILES = {
    'v1': ['test.jsonl'],
    'v2': ['test2.jsonl'],
    'v3': ['test3.jsonl'],
    'v4': ['test4.jsonl'],
    'v5': ['test5.jsonl'],
    'v6': ['test6.jsonl'],
    'release_v1': ['test.jsonl'],
    'release_v2': ['test.jsonl', 'test2.jsonl'],
    'release_v3': ['test.jsonl', 'test2.jsonl', 'test3.jsonl'],
    'release_v4': ['test.jsonl', 'test2.jsonl', 'test3.jsonl', 'test4.jsonl'],
    'release_v5': ['test.jsonl', 'test2.jsonl', 'test3.jsonl', 'test4.jsonl', 'test5.jsonl'],
    'release_v6': ['test.jsonl', 'test2.jsonl', 'test3.jsonl', 'test4.jsonl', 'test5.jsonl', 'test6.jsonl'],
    'release_latest': ['test.jsonl', 'test2.jsonl', 'test3.jsonl', 'test4.jsonl', 'test5.jsonl', 'test6.jsonl'],
}


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench',
        pretty_name='Live-Code-Bench',
        tags=[Tags.CODING],
        description=
        'Live Code Bench is a benchmark for evaluating code generation models on real-world coding tasks. It includes a variety of programming problems with test cases to assess the model\'s ability to generate correct and efficient code solutions. '
        '**By default the code is executed in local environment. We recommend using sandbox execution to safely run and evaluate the generated code, please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for more details.**',
        dataset_id=_LCB_LOCAL_PATH,
        subset_list=['easy', 'medium', 'hard'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template=
        '### Question:\n{question_content}\n\n{format_prompt} ### Answer: (use the provided format with backticks)\n\n',
        review_timeout=6,
        extra_params={
            'version': {
                'type': 'str',
                'description': (
                    '要测试的 LiveCodeBench 版本。单版本: v1~v6；累积版本: release_v1~release_v6 / release_latest。'
                    '例如 v2 仅测试第 2 批题目，release_latest 测试全部题目。'
                ),
                'value': 'release_latest',
                'choices': list(_LCB_VERSION_FILES.keys()),
            },
            'start_date': {
                'type': 'str | null',
                'description': 'Filter problems starting from this date (YYYY-MM-DD). Null keeps all.',
                'value': None
            },
            'end_date': {
                'type': 'str | null',
                'description': 'Filter problems up to this date (YYYY-MM-DD). Null keeps all.',
                'value': None
            },
            'debug': {
                'type': 'bool',
                'description': 'Enable verbose debug logging and bypass certain safety checks.',
                'value': False
            }
        },
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {}
            }
        },
    )
)
class LiveCodeBenchAdapter(DefaultDataAdapter):
    """
    Live Code Bench adapter using the new data processing framework.
    支持通过 test_n 参数自由选择版本（v1~v6 / release_latest 等），
    并按 difficulty 字段（easy / medium / hard）分类统计结果。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.version = self.extra_params.get('version', 'release_latest')
        self.debug = self.extra_params.get('debug', False)
        self.start_date = self.extra_params.get('start_date')
        self.end_date = self.extra_params.get('end_date')

        self.save_metadata = False  # Don't save metadata, since they are large

    def load_dataset(self) -> DatasetDict:
        """直接从本地 JSONL 文件加载数据，按 difficulty 分组后返回 DatasetDict。"""
        from evalscope.api.dataset import MemoryDataset
        from .load_utils import filter_date

        files = _LCB_VERSION_FILES.get(self.version, _LCB_VERSION_FILES['release_latest'])
        logger.info(f'[LiveCodeBench] 加载版本 {self.version}，文件列表: {files}')

        all_records = []
        for fname in files:
            fpath = os.path.join(_LCB_LOCAL_PATH, fname)
            records = jsonl_to_list(fpath)
            all_records.extend(records)
        logger.info(f'[LiveCodeBench] 共加载 {len(all_records)} 条样本')

        # 转换为 Sample，按 difficulty 分组
        samples_by_diff: Dict[str, list] = {'easy': [], 'medium': [], 'hard': []}
        for record in all_records:
            # 日期过滤（仅在设置了日期范围且 contest_date 非空时执行）
            contest_date = record.get('contest_date', '')
            if (self.start_date or self.end_date) and contest_date:
                if not filter_date(contest_date, start_date=self.start_date, end_date=self.end_date):
                    continue

            sample = self.record_to_sample(record)
            diff = sample.subset_key or 'unknown'
            if diff in samples_by_diff:
                samples_by_diff[diff].append(sample)

        # 应用 limit（按 easy/medium/hard 顺序依次截断，总数不超过 limit）
        if self.limit:
            limit = int(self.limit)
            remaining = limit
            for diff in list(samples_by_diff.keys()):
                take = min(remaining, len(samples_by_diff[diff]))
                samples_by_diff[diff] = samples_by_diff[diff][:take]
                remaining -= take

        # 构建 DatasetDict
        dataset_dict_data = {}
        for diff, diff_samples in samples_by_diff.items():
            if not diff_samples:
                continue
            ds = MemoryDataset(diff_samples, name=f'live_code_bench_{diff}')
            ds.reindex(group_size=self.repeats)
            dataset_dict_data[diff] = ds

        counts = {d: len(s) for d, s in samples_by_diff.items()}
        logger.info(f'[LiveCodeBench] 各难度样本数: {counts}')

        self.test_dataset = DatasetDict(dataset_dict_data)
        self.fewshot_dataset = None
        return self.test_dataset

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        from .load_utils import transform

        record = transform(record)

        question_content = record['question_content']
        format_prompt = record['format_prompt']
        full_prompt = self.prompt_template.format(question_content=question_content, format_prompt=format_prompt)

        return Sample(
            input=[ChatMessageUser(content=full_prompt)],
            target='',
            subset_key=record.get('difficulty'),  # 用于按难度分组统计
            metadata={
                'evaluation_sample': record['evaluation_sample'],
                'contest_date': record['contest_date'],
                'difficulty': record.get('difficulty', 'unknown'),
            }
        )

    def sample_filter(self, sample):
        # 日期过滤已在 load_dataset 中处理，此处直接放行
        return True

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract code from the prediction."""
        from .extract_utils import extract_code_generation
        return extract_code_generation(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        if not self.use_sandbox:
            # Use original evaluation method
            from .evaluate_utils import codegen_metrics

            references = [{'input_output': task_state.metadata['evaluation_sample']}]
            predictions = [[filtered_prediction]]

            try:
                metrics, eval_results, final_metadata = codegen_metrics(
                    references,
                    predictions,
                    k_list=[1],
                    num_process_evaluate=1,
                    timeout=self.review_timeout,
                    debug=self.debug,
                )
                pass_rate = metrics['pass@1'] / 100  # convert to point scale

                score.value = {'acc': float(pass_rate > 0)}
                score.explanation = f"Pass@1: {metrics['pass@1']}%"

                # Convert numpy types to native Python types for JSON serialization
                serializable_eval_results = convert_normal_types(eval_results)
                serializable_final_metadata = convert_normal_types(final_metadata)

                score.metadata = {
                    'pass_rate': float(pass_rate),
                    'timeout': self.review_timeout,
                    'debug': self.debug,
                    'eval_results': serializable_eval_results,
                    'final_metadata': serializable_final_metadata
                }
            except Exception as e:
                score.value = {'acc': False}
                score.explanation = f'Evaluation failed: {str(e)}'
                score.metadata = {'error': str(e)}
        else:
            # Use sandbox execution
            try:
                from .sandbox_evaluate_utils import evaluate_in_sandbox

                evaluation_sample = task_state.metadata['evaluation_sample']
                passed, detailed_results = evaluate_in_sandbox(
                    self, filtered_prediction, evaluation_sample, timeout=self.review_timeout, debug=self.debug
                )

                score.value = {'acc': passed}
                score.explanation = f"Sandbox execution: {'Passed' if passed else 'Failed'}"
                score.metadata = {
                    'timeout': self.review_timeout,
                    'debug': self.debug,
                    'execution_method': 'sandbox',
                    'detailed_results': detailed_results
                }
            except Exception as e:
                score.value = {'acc': False}
                score.explanation = f'Sandbox evaluation failed: {str(e)}'
                score.metadata = {'error': str(e), 'execution_method': 'sandbox'}

        score.main_score_name = 'acc'
        return score
