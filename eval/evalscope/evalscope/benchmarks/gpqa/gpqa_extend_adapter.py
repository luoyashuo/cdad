import os
import random
import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import LocalDataLoader, Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

LOCAL_CSV_PATH = '/data/modelscope_cache/datasets/gpqa_extended.csv'


@register_benchmark(
    BenchmarkMeta(
        name='gpqa_extend',
        pretty_name='GPQA-Extended',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description='GPQA Extended dataset for evaluating reasoning on graduate-level science problems.',
        dataset_id='modelscope/gpqa',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class GPQAExtendAdapter(MultiChoiceAdapter):

    def load(self):
        if os.path.exists(LOCAL_CSV_PATH):
            from evalscope.api.dataset import DatasetDict, LocalDataLoader
            loader = LocalDataLoader(
                data_id_or_path=LOCAL_CSV_PATH,
                split=self.eval_split,
                subset=self.default_subset,
                sample_fields=self.record_to_sample,
                filter_func=self.sample_filter,
                limit=self.limit,
                repeats=self.repeats,
                shuffle=self.shuffle,
                shuffle_choices=self.shuffle_choices,
            )
            dataset = loader.load()
            return DatasetDict({self.default_subset: dataset}), None
        else:
            return self.load_from_remote()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        processed_data = self._process_input(record)
        return Sample(
            input=record['Question'],
            choices=processed_data['choices'],
            target=processed_data['answer'],
            subset_key=record.get('Subdomain', record.get('subset', '')),
            metadata={
                'correct_answer': record['Correct Answer'],
                'incorrect_answers': [
                    record['Incorrect Answer 1'],
                    record['Incorrect Answer 2'],
                    record['Incorrect Answer 3'],
                ],
            },
        )

    def _process_input(self, input_d: dict) -> dict:
        def preprocess(text):
            if text is None:
                return ' '
            text = text.strip()
            text = text.replace(' [title]', '. ')
            text = re.sub('\\[.*?\\]', '', text)
            text = text.replace('  ', ' ')
            return text

        choices = [
            preprocess(input_d['Incorrect Answer 1']),
            preprocess(input_d['Incorrect Answer 2']),
            preprocess(input_d['Incorrect Answer 3']),
            preprocess(input_d['Correct Answer']),
        ]
        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(input_d['Correct Answer']))
        return {
            'choices': choices,
            'answer': f'{chr(65 + correct_answer_index)}',
        }
