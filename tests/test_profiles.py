import json
import os
import subprocess
import sys

import pytest

from ssrb.profiles import evaluate_profile, expected_profile


@pytest.mark.parametrize("gpu", [None, "NVIDIA A40", "NVIDIA A100-SXM4-80GB"])
def test_reference_preflight_rejects_other_gpus(gpu):
    actual = expected_profile("lidc-dino-reference")
    actual["gpu"] = gpu
    result = evaluate_profile("lidc-dino-reference", actual)
    assert result["status"] == "different-execution-profile"
    assert result["differences"]["gpu"]["actual"] == gpu


@pytest.mark.parametrize("key,value", [
    ("cpu_threads", 4), ("torch", "2.9.1+cpu"), ("matmul_allow_tf32", True),
    ("cudnn_deterministic", False), ("mkl_threads", None),
])
def test_reference_preflight_checks_arithmetic_and_runtime(key, value):
    actual = expected_profile("lidc-dino-reference")
    actual[key] = value
    assert key in evaluate_profile("lidc-dino-reference", actual)["differences"]


def test_wider_profile_does_not_certify_paper_trajectory():
    result = evaluate_profile("wide-repeatable", expected_profile("wide-repeatable"))
    assert result["status"] == "verified-execution-profile"
    assert "does not reconstruct" in result["scope"]
    with pytest.raises(ValueError, match="Unknown training profile"):
        expected_profile("all-paper-results-exact")


def test_cpu_cli_fails_before_expensive_training():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    run = subprocess.run([sys.executable, "-I", "-m", "ssrb.cli", "check-profile", "--profile", "lidc-dino-reference"],
                         env=env, capture_output=True, text=True, check=False)
    assert run.returncode == 1, run.stderr
    assert run.stdout.strip(), run.stderr
    result = json.loads(run.stdout)
    assert result["status"] == "different-execution-profile"
    assert result["differences"]["gpu"]["actual"] is None
