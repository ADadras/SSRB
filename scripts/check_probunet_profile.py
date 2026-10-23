"""Check the validated Prob-U execution settings before starting training."""
import json
import hashlib
import importlib.metadata as metadata
import subprocess
import os
from pathlib import Path
import sys

import numpy as np
import PIL
import torch
from ssrb.execution import configure, record

reference = json.loads((Path(__file__).resolve().parents[1] / 'artifacts/probunet_reference_execution.json').read_text())
configure(True)
actual = record()
expected = {key: reference[key] for key in (
    'gpu', 'python', 'torch', 'deterministic_algorithms', 'cudnn_deterministic',
    'cudnn_benchmark', 'cudnn_allow_tf32', 'matmul_allow_tf32', 'cublas_workspace_config')}
expected.update(cuda_version=reference['cuda'], cudnn_version=reference['cudnn'],
                cpu_threads=reference['omp_threads'], omp_threads=str(reference['omp_threads']),
                mkl_threads=str(reference['mkl_threads']), pillow='11.3.0', numpy='2.2.6')
actual.update(cpu_threads=torch.get_num_threads(), omp_threads=os.environ.get('OMP_NUM_THREADS'),
              mkl_threads=os.environ.get('MKL_NUM_THREADS'), pillow=PIL.__version__, numpy=np.__version__)
# Record the driver and actual CUDA-library bytes, rather than relying only
# on version labels from an environment that may have been reinstalled.
try:
    drivers = sorted(set(subprocess.check_output(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
        text=True, stderr=subprocess.DEVNULL).split()))
except (OSError, subprocess.CalledProcessError):
    drivers = []
libraries = {}
for relative in reference['cuda_library_sha256']:
    component, filename = relative.split('/')
    try:
        distribution = metadata.distribution('nvidia-' + component + '-cu12')
    except metadata.PackageNotFoundError:
        continue
    path = Path(distribution.locate_file('nvidia/' + component + '/lib/' + filename))
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                digest.update(block)
        libraries[relative] = digest.hexdigest()
expected.update(driver_versions=reference['driver_versions'], cuda_library_sha256=reference['cuda_library_sha256'],
                compute_capability=reference['compute_capability'])
actual.update(driver_versions=drivers, cuda_library_sha256=libraries,
              compute_capability=list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None)
differences = {key: {'expected': value, 'actual': actual[key]} for key, value in expected.items()
               if actual[key] != value}
print(json.dumps(dict(status='different-execution-profile' if differences else 'verified-probunet-reference-profile',
                      differences=differences, actual=actual), indent=2))
sys.exit(1 if differences else 0)
