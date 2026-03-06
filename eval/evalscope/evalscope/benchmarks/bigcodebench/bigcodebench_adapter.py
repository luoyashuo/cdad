import multiprocessing
import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

INSTRUCT_TEMPLATE = (
    'Please write a Python function to complete the following task:\n\n'
    '{instruct_prompt}\n\n'
    'Enclose your implementation in a Python code block:\n'
    '```python\n'
    '# your implementation here\n'
    '```'
)

COMPLETE_TEMPLATE = (
    'Complete the following Python function. '
    'Return only the completed function in a code block.\n\n'
    '```python\n'
    '{complete_prompt}\n'
    '```'
)


def _unsafe_execute(code: str, timeout: float, result: list):
    """Execute BigCodeBench unittest in a subprocess with timeout."""
    import signal
    import unittest

    class _Timeout(Exception):
        pass

    def _handler(signum, frame):
        raise _Timeout()

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout) + 2)
    try:
        exec_globals = {}
        exec(code, exec_globals)  # noqa: S102

        if 'TestCases' not in exec_globals:
            result.append({'passed': False, 'result': 'TestCases class not found'})
            return

        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(exec_globals['TestCases'])
        test_result = unittest.TestResult()
        suite.run(test_result)

        if test_result.wasSuccessful():
            result.append({'passed': True, 'result': 'passed'})
        else:
            errs = [str(e[1]) for e in test_result.failures + test_result.errors]
            result.append({'passed': False, 'result': f'failed: {errs[0][:300] if errs else "unknown"}'})
    except _Timeout:
        result.append({'passed': False, 'result': 'timed out'})
    except Exception as e:
        result.append({'passed': False, 'result': f'{type(e).__name__}: {e}'})
    finally:
        signal.alarm(0)


def _run_test_locally(code: str, timeout: float) -> dict:
    """Run BigCodeBench test in a child process and return result."""
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_unsafe_execute, args=(code, timeout, result))
    p.start()
    p.join(timeout=timeout + 3)
    if p.is_alive():
        p.kill()
        p.join()
    if not result:
        return {'passed': False, 'result': 'timed out'}
    return result[0]


@register_benchmark(
    BenchmarkMeta(
        name='bigcodebench',
        pretty_name='BigCodeBench',
        tags=[Tags.CODING],
        description=(
            'BigCodeBench evaluates code generation with diverse function calls and complex instructions. '
            'Contains 1140 Python programming tasks with unittest test cases covering a wide range of libraries. '
            'Supports instruct mode (NL-based) and complete mode (code completion). '
            'Local execution without sandbox is supported (code runs in subprocess).'
        ),
        dataset_id='bigcode/bigcodebench',
        subset_list=['default'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='v0.1.4',
        prompt_template='',  # built dynamically per mode in record_to_sample
        review_timeout=20,
        extra_params={
            'mode': {
                'type': 'str',
                'description': 'Evaluation variant: "instruct" (NL prompt) or "complete" (code completion)',
                'value': 'instruct',
                'choices': ['instruct', 'complete'],
            }
        },
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {},
            },
        },
    )
)
class BigCodeBenchAdapter(DefaultDataAdapter):
    """
    BigCodeBench adapter.

    Supports two modes via extra_params['mode']:
      - instruct : model receives a NL task description and generates a complete function
      - complete : model receives the function signature + docstring and fills in the body

    Local execution (use_sandbox=false) runs the unittest in a child process via multiprocessing.
    """

    def load(self):
        """Override to force HuggingFace hub (dataset lives in HF_HOME cache)."""
        with self._temporary_attribute('dataset_hub', HubType.HUGGINGFACE):
            return self.load_from_remote()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        mode = self.extra_params.get('mode', 'instruct')
        if mode == 'complete':
            full_prompt = COMPLETE_TEMPLATE.format(complete_prompt=record['complete_prompt'])
        else:
            full_prompt = INSTRUCT_TEMPLATE.format(instruct_prompt=record['instruct_prompt'])

        return Sample(
            input=[ChatMessageUser(content=full_prompt)],
            target=record['canonical_solution'],
            metadata={
                'task_id': record['task_id'],
                'mode': mode,
                'code_prompt': record['code_prompt'],      # imports + function signature
                'complete_prompt': record['complete_prompt'],
                'test': record['test'],
                'entry_point': record['entry_point'],
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return _extract_code(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        meta = task_state.metadata

        exec_code = _build_exec_code(
            mode=meta['mode'],
            code_prompt=meta['code_prompt'],
            complete_prompt=meta['complete_prompt'],
            completion=filtered_prediction,
            test=meta['test'],
        )

        if not self.use_sandbox:
            res = _run_test_locally(exec_code, timeout=self.review_timeout)
            passed = res['passed']
        else:
            res = self.execute_code_in_sandbox(code=exec_code, timeout=self.review_timeout, language='python')
            passed = res.get('status') == 'success'

        score.value = {'acc': passed}
        score.metadata = {
            'task_id': meta['task_id'],
            'mode': meta['mode'],
            'timeout': self.review_timeout,
            'execution_result': res,
        }
        score.main_score_name = 'acc'
        return score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_code(text: str) -> str:
    """Extract the first Python code block from markdown, or return raw text."""
    blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    return text.strip()


def _build_exec_code(mode: str, code_prompt: str, complete_prompt: str, completion: str, test: str) -> str:
    """
    Build the full code to execute for evaluation.

    - instruct: model generates a self-contained function → run completion + test directly.
    - complete: model continues from the function signature/docstring → prepend complete_prompt.
    """
    if mode == 'complete':
        # complete_prompt ends with the function signature (no body).
        # The model fills in the body starting from the next line.
        exec_code = complete_prompt + '\n' + completion
    else:
        # instruct: the extracted code should be self-contained (includes imports + def).
        exec_code = completion

    return exec_code + '\n\n' + test
