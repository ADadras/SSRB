"""Linux Landlock read restrictions for a separate trainer process.

API reference: https://docs.kernel.org/userspace-api/landlock.html
The handled rights are READ_FILE and READ_DIR. This limits data reads; it is
not a general sandbox for arbitrary untrusted programs or network activity.
"""
import ctypes
import os
import platform
import sys
from pathlib import Path


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class Beneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def restrict_reads(paths):
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("read isolation requires Linux x86_64/aarch64 with Landlock")
    libc = ctypes.CDLL(None, use_errno=True)
    def call(number, *args):
        result = libc.syscall(number, *args)
        if result < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return result
    abi = call(444, 0, 0, 1)
    rights = (1 << 2) | (1 << 3)
    attr = Ruleset(rights)
    fd = call(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    try:
        for name in sorted({str(Path(p).resolve()) for p in paths if Path(p).exists()}):
            path = Path(name)
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Beneath(rights if path.is_dir() else (1 << 2), parent)
                call(445, fd, 1, ctypes.byref(rule), 0)
            finally:
                os.close(parent)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot set no_new_privs")
        call(446, fd, 0)
    finally:
        os.close(fd)
    return abi


def isolate_trainer(root, manifest, output, config=None, anchor=None):
    root, manifest, output = Path(root).resolve(), Path(manifest).resolve(), Path(output).resolve()
    if not manifest.is_relative_to(root):
        raise ValueError("the visible manifest must be inside the trainer root")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    import tempfile
    import atexit
    import shutil
    scratch = Path(tempfile.mkdtemp(prefix="ssrb-", dir="/var/tmp" if os.access("/var/tmp", os.W_OK) else "/tmp"))
    owner = os.getpid()
    atexit.register(lambda: shutil.rmtree(scratch, ignore_errors=True) if os.getpid() == owner else None)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["CUDA_CACHE_PATH"] = str(scratch/"cuda")
    tempfile.tempdir = str(scratch)
    paths = [root, output, scratch, sys.prefix, sys.base_prefix, Path(__file__).resolve().parent,
             "/usr", "/lib", "/lib64", "/etc", "/dev", "/proc", "/sys"]
    paths.extend(p for p in (config, anchor) if p)
    return restrict_reads(paths)
