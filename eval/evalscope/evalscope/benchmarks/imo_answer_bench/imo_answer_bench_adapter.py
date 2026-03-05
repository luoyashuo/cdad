from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, Tags


@register_benchmark(
    BenchmarkMeta(
        name='imo_answer_bench',
        pretty_name='IMO-AnswerBench',
        tags=[Tags.MATH, Tags.REASONING],
        description='IMO-AnswerBench contains 400 competition-level math problems (100 each from '
                    'Algebra, Combinatorics, Geometry, Number Theory) sourced from IMO shortlists '
                    'and similar olympiad competitions. Each problem has a short numerical/symbolic answer.',
        dataset_id='OpenEvals/IMO-AnswerBench',
        subset_list=['default'],
        metric_list=[{'acc': {'numeric': True}}],
        eval_split='train',
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
        system_prompt='Please reason step by step to solve the problem. Put your final answer in \\boxed{}.',
    )
)
class IMOAnswerBenchAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._use_llm_judge = False

    def load(self):
        with self._temporary_attribute('dataset_hub', HubType.HUGGINGFACE):
            return self.load_from_remote()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['Problem'],
            target=record['Short Answer'],
            metadata={
                'problem_id': record['Problem ID'],
                'category': record['Category'],
                'subcategory': record['Subcategory'],
                'source': record['Source'],
            },
        )

    def extract_answer(self, prediction: str, task_state) -> str:
        from evalscope.metrics.math_parser import extract_answer
        return extract_answer(prediction)
