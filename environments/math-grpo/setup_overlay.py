"""Prepare or install an AutoDL overlay without replacing inherited GPU packages."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def protected(name):
    name = name.lower().replace('_', '-')
    return name in {'torch', 'torchvision', 'torchaudio', 'triton'} or name.startswith(('nvidia-', 'cuda-'))


def reject_gpu_changes(report):
    names = [p['metadata']['name'] for p in report['install'] if protected(p['metadata']['name'])]
    if names:
        raise RuntimeError(f'Refusing to install or replace image GPU packages: {names}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install', action='store_true', help='Install after validating the dependency plan')
    p.add_argument('--index-url', default='https://pypi.tuna.tsinghua.edu.cn/simple')
    args = p.parse_args()
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('Run inside the AutoDL Linux Python 3.12 overlay')
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Create and activate the system-site-packages venv first')
    cfg = Path(sys.prefix)/'pyvenv.cfg'
    if 'include-system-site-packages = true' not in cfg.read_text().lower():
        raise RuntimeError('This environment must inherit image site-packages')
    import torch
    if torch.__version__.split('+')[0] != '2.8.0' or torch.version.cuda != '12.8':
        raise RuntimeError(f'Expected image Torch 2.8.0 / CUDA 12.8, got {torch.__version__}/{torch.version.cuda}')
    if Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('Torch is installed in the overlay instead of inherited from the image')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA/BF16 unavailable')
    # Record the complete image+overlay inventory; constrain GPU packages exactly.
    inventory = {d.metadata['Name']: d.version for d in metadata.distributions()}
    audit = Path(sys.prefix)/'grpo_environment_audit'
    audit.mkdir(exist_ok=True)
    (audit/'before.json').write_text(json.dumps(dict(python=sys.version, executable=sys.executable,
        torch_file=torch.__file__, packages=inventory), indent=2))
    constraints = audit/'image-gpu-constraints.txt'
    constraints.write_text(''.join(f'{n}=={v}\n' for n,v in inventory.items() if protected(n)))
    report_path = audit/'install-plan.json'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--dry-run', '--report', str(report_path),
        '--index-url', args.index_url, '-c', str(constraints), '-r', str(HERE/'requirements-overlay.txt')], check=True)
    report = json.loads(report_path.read_text())
    reject_gpu_changes(report)
    print('Plan validated: no Torch/CUDA package installs. Audit:', audit, flush=True)
    if not args.install:
        print('Preview only. Rerun with --install to apply.')
        return
    # Install ONLY the vetted list. No second dependency resolution can introduce GPU packages.
    requested = []
    for item in report['install']:
        name, version = item['metadata']['name'], item['metadata']['version']
        if name.lower() == 'latex2sympy2':
            requested.append(next(line for line in (HERE/'requirements-overlay.txt').read_text().splitlines()
                                  if line.startswith('latex2sympy2 @')))
        else:
            requested.append(f'{name}=={version}')
    if requested:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                        '--index-url', args.index_url, *requested], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    subprocess.run([sys.executable, '-c',
        "import torch; from trl import GRPOTrainer; import peft; "
        "x=torch.randn(256,256,device='cuda',dtype=torch.bfloat16,requires_grad=True); "
        "(x@x).float().square().mean().backward(); torch.cuda.synchronize(); "
        "print('TRL import and GPU BF16 backward OK; Torch:',torch.__version__,torch.__file__)"], check=True)
    (audit/'installed-freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze', '--all'], text=True))


if __name__ == '__main__':
    main()
