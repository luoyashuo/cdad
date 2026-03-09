from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

USER_PROMPT_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

Question:
{question}
Options:
{choices}
""".lstrip()

SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE = """
The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with 'ANSWER: [LETTER]' (without quotes) where [LETTER] is the correct letter choice.

{examples}
""".lstrip()


@register_benchmark(
    BenchmarkMeta(
        name='mmlu_pro_no_math',
        pretty_name='MMLU-Pro (No Math)',
        tags=[Tags.MULTIPLE_CHOICE, Tags.KNOWLEDGE],
        description='MMLU-Pro benchmark excluding the math category.',
        dataset_id='TIGER-Lab/MMLU-Pro',
        subset_list=['default'],
        metric_list=['acc'],
        few_shot_num=5,
        train_split='validation',
        eval_split='test',
        prompt_template=USER_PROMPT_TEMPLATE,
        few_shot_prompt_template=SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE + USER_PROMPT_TEMPLATE,
    )
)
class MMLUProNoMathAdapter(MultiChoiceAdapter):

    def load(self):
        with self._temporary_attribute('dataset_hub', HubType.HUGGINGFACE):
            return self.load_from_remote()

    def sample_filter(self, sample: Sample) -> bool:
        return sample.metadata.get('subject', '') != 'math'

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['options'],
            target=record['answer'],
            metadata={
                'cot_content': record['cot_content'],
                'subject': record['category'].lower(),
                'question_id': record['question_id'],
            },
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        q_str = f'Question:\n{str(sample.input)}'
        options = sample.choices if sample.choices is not None else []
        opt_str = 'Options:\n' + '\n'.join(f'{chr(65 + i)} {opt}' for i, opt in enumerate(options))
        ans_str = sample.metadata['cot_content'] if sample.metadata is not None else ''
        ans_str = ans_str.replace('The answer is', 'ANSWER:')
        ans_opt = ans_str.split('ANSWER:')[-1].split('.')[0].strip().strip('(').strip(')')
        ans_str = ans_str.replace(f'ANSWER: ({ans_opt})', f'ANSWER: {ans_opt}')
        return '\n'.join([q_str, opt_str, ans_str])

    def format_fewshot_template(self, fewshot, sample):
        fewshot_str = SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE.format(
            subject=sample.metadata['subject'],
            examples=fewshot,
        )
        return fewshot_str + '\n' + self.format_prompt_template(sample)
