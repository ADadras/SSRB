"""Optional deterministic execution and the numerical settings recorded per fit."""
import os
import platform
from pathlib import Path
import torch


def configure(deterministic=False):
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def record():
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                cpu_model=_cpu_model(), cpu_architecture=platform.machine(),
                cpu_capability=torch.backends.cpu.get_cpu_capability(),
                cpu_threads=torch.get_num_threads(),
                cpu_interop_threads=torch.get_num_interop_threads(),
                aten_cpu_capability_override=os.environ.get('ATEN_CPU_CAPABILITY'),
                mkl_cbwr=os.environ.get('MKL_CBWR'),
                mkl_enable_instructions=os.environ.get('MKL_ENABLE_INSTRUCTIONS'),
                cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
