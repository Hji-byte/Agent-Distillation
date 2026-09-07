import pytest
import json
from scripts.training.train_cot_pair import encode_row, PROFILES


class FakeTokenizer:
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['enable_thinking'] is False and kwargs['truncation'] is False
        return [1,2,3] if kwargs['add_generation_prompt'] else [1,2,3,4,5,6]


def row(content='Answer'):
    return {'question_id':'q','messages':[{'role':'system','content':'s'},{'role':'user','content':'q'},{'role':'assistant','content':content}]}


def test_mask_and_inclusive_cap():
    result=encode_row(FakeTokenizer(),row(),6)
    assert result['labels']==[-100,-100,-100,4,5,6]
    with pytest.raises(ValueError):
        encode_row(FakeTokenizer(),row(),5)


def test_think_and_bad_prefix_rejected():
    with pytest.raises(ValueError):
        encode_row(FakeTokenizer(),row('</think>answer'))
    class BadTokenizer(FakeTokenizer):
        def apply_chat_template(self,messages,**kwargs):
            return [9] if kwargs['add_generation_prompt'] else [1,2]
    with pytest.raises(ValueError):
        encode_row(BadTokenizer(),row())


@pytest.mark.parametrize('cap', [4096, 6400])
def test_profile_matches_saved_pairs(cap):
    path, count = PROFILES[cap]
    summary = json.loads((path/'summary.json').read_text(encoding='utf-8'))
    assert summary['max_length'] == cap
    assert summary['kept_pairs'] == count
    groups = [[json.loads(line) for line in (path/f'{name}_correct_sft.jsonl').read_text(encoding='utf-8').splitlines()]
              for name in ('normal', 'shortest')]
    assert len(groups[0]) == len(groups[1]) == count
    assert [r['question_id'] for r in groups[0]] == [r['question_id'] for r in groups[1]]


def test_real_4096_boundary():
    class LongTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [1,2,3] if kwargs['add_generation_prompt'] else [1,2,3]+[4]*4093
    assert len(encode_row(LongTokenizer(), row(), 4096)['input_ids']) == 4096
    with pytest.raises(ValueError):
        encode_row(LongTokenizer(), row(), 4095)
