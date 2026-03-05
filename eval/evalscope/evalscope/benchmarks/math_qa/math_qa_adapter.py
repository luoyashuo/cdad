from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = (
    'MathQA dataset is gathered by using a new representation language to annotate over the '
    'AQuA-RAT dataset with fully-specified operational programs.'
)

BOXED_ANSWER_MCQ_TEMPLATE = (
    'Answer the following multiple choice question. '
    'Think step by step before answering. '
    'Put the letter of your final answer (from {letters}) within \\boxed{{}}, '
    'e.g., \\boxed{{[LETTER]}}.\n\n'
    '{question}\n\n'
    '{choices}'
)


@register_benchmark(
    BenchmarkMeta(
        name='math_qa',
        pretty_name='MathQA',
        tags=[Tags.REASONING, Tags.MATH, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/math-qa',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=BOXED_ANSWER_MCQ_TEMPLATE,
        system_prompt='Please reason step by step to solve the problem. Put the letter of your final answer (A, B, C, D, or E) in \\boxed{}, e.g., \\boxed{[LETTER]}.',
    )
)
class MathQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={'reasoning': record['reasoning']},
        )

    def extract_answer(self, prediction: str, task_state) -> str:
        from evalscope.metrics.math_parser import extract_answer
        return extract_answer(prediction)
