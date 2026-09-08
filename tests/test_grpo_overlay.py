import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('grpo_overlay',
    Path(__file__).resolve().parents[1]/'environments/math-grpo/setup_overlay.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('name', ['torch','torchvision','torchaudio','triton','nvidia-cublas-cu12','cuda-python'])
def test_reject_gpu_changes(name):
    with pytest.raises(RuntimeError):
        module.reject_gpu_changes({'install':[{'metadata':{'name':name}}]})


def test_allow_training_dependencies():
    module.reject_gpu_changes({'install':[{'metadata':{'name':'trl'}}]})
    module.reject_gpu_changes({'install':[]})
