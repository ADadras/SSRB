"""Preflight checks for the training setups validated with the released code."""
import os

import numpy as np
import PIL
import scipy

from .execution import configure, record


PROFILE_NAMES = ("lidc-dino-reference", "wide-repeatable")


def expected_profile(name):
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown training profile: {name}")
    return dict(
        gpu="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        cpu_model="AMD EPYC 9475F 48-Core Processor",
        cpu_threads=8, omp_threads="8", mkl_threads="8",
        python="3.12.12", torch="2.9.1+cu128", cuda_version="12.8",
        cudnn_version=91002, numpy="2.2.6", scipy="1.16.3", pillow="11.3.0",
        deterministic_algorithms=True, cudnn_deterministic=True,
        cudnn_benchmark=False, matmul_allow_tf32=False, cudnn_allow_tf32=True,
        cublas_workspace_config=":4096:8",
    )


def evaluate_profile(name, actual):
    expected = expected_profile(name)
    differences = {key: dict(expected=value, actual=actual.get(key))
                   for key, value in expected.items() if actual.get(key) != value}
    scope = (
        "Recovers the tested paper LIDC RN/Capacity trajectories and hard-MLQ comparison; "
        "does not certify every task or metric."
        if name == "lidc-dino-reference" else
        "Repeatable wider-model training; "
        "does not reconstruct the paper's nondeterministic wider-model trajectory."
    )
    return dict(profile=name, status="different-execution-profile" if differences else "verified-execution-profile",
                differences=differences, actual=actual, scope=scope,
                training_instruction="This check configures only its own process. Export the thread/workspace "
                                     "variables and pass --deterministic to each training command.")


def check_profile(name):
    expected_profile(name)
    configure(True)
    actual = record()
    actual.update(omp_threads=os.environ.get("OMP_NUM_THREADS"), mkl_threads=os.environ.get("MKL_NUM_THREADS"),
                  numpy=np.__version__, scipy=scipy.__version__, pillow=PIL.__version__)
    return evaluate_profile(name, actual)
