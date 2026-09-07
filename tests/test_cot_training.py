import pytest
from scripts.training.train_cot_pair import encode_row


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
