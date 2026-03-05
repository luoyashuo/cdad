from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, Tags


def _clean_query(query: str) -> str:
    """Strip 'Q: ' prefix and '\\nA: The answer is' suffix from the raw query field."""
    q = query.strip()
    if q.startswith('Q: '):
        q = q[3:]
    suffix = '\nA: The answer is'
    if q.endswith(suffix):
        q = q[:-len(suffix)]
    return q.strip()


@register_benchmark(
    BenchmarkMeta(
        name='agieval_math',
        pretty_name='AGIEval-Math',
        tags=[Tags.MATH, Tags.REASONING],
        description='AGIEval-Math is a subset of AGIEval containing 1000 competition-level math '
                    'problems drawn from the MATH dataset, covering algebra, geometry, number theory '
                    'and more. Answers are clean numerical or symbolic expressions.',
        dataset_id='hails/agieval-math',
        subset_list=['default'],
        metric_list=[{'acc': {'numeric': True}}],
        eval_split='test',
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
        system_prompt='Please reason step by step to solve the problem. Put your final answer in \\boxed{}.',
    )
)
class AGIEvalMathAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._use_llm_judge = False

    def load(self):
        with self._temporary_attribute('dataset_hub', HubType.HUGGINGFACE):
            return self.load_from_remote()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=_clean_query(record['query']),
            target=record['answer'],
        )

    def extract_answer(self, prediction: str, task_state) -> str:
        from evalscope.metrics.math_parser import extract_answer
        return extract_answer(prediction)
