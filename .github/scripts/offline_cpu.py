"""Run a CPU test with internet sockets disabled for the process and its children."""

import ctypes
import ctypes.util
import errno
import os
import socket
import sys


# Values from libseccomp's seccomp.h.
SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_ERRNO = 0x00050000
SCMP_CMP_EQ = 4


class ArgumentComparison(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def block_internet_sockets() -> None:
    library = ctypes.CDLL(ctypes.util.find_library("seccomp"), use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ArgumentComparison),
    ]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_release.restype = None

    context = library.seccomp_init(SCMP_ACT_ALLOW)
    if not context:
        raise RuntimeError("seccomp initialization failed")
    families = (socket.AF_INET, socket.AF_INET6)
    try:
        socket_syscall = library.seccomp_syscall_resolve_name(b"socket")
        for family in families:
            comparison = ArgumentComparison(0, SCMP_CMP_EQ, family, 0)
            status = library.seccomp_rule_add_array(
                context, SCMP_ACT_ERRNO | errno.EPERM, socket_syscall, 1,
                ctypes.byref(comparison),
            )
            if status != 0:
                raise RuntimeError("network guard setup failed")
        if library.seccomp_load(context) != 0:
            raise RuntimeError("network guard load failed")
    finally:
        library.seccomp_release(context)

    for family in families:
        try:
            with socket.socket(family, socket.SOCK_STREAM):
                raise RuntimeError("network guard verification failed")
        except PermissionError:
            pass


if __name__ == "__main__":
    block_internet_sockets()
    os.environ.update(
        CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none", ROCR_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        WANDB_MODE="offline", PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    )
    os.execvp(sys.argv[1], sys.argv[1:])
