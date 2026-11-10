"""Restrict this test process and its children to CPU and offline operation."""
import ctypes
import ctypes.util
import errno
import os
import socket
import sys

lib = ctypes.CDLL(ctypes.util.find_library("seccomp"), use_errno=True)
class Comparison(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]
lib.seccomp_init.argtypes = [ctypes.c_uint32]
lib.seccomp_init.restype = ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                      ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Comparison)]
lib.seccomp_load.argtypes = [ctypes.c_void_p]
lib.seccomp_release.argtypes = [ctypes.c_void_p]
context = lib.seccomp_init(0x7fff0000)
if not context:
    raise RuntimeError("seccomp initialization failed")
syscall = lib.seccomp_syscall_resolve_name(b"socket")
for family in (socket.AF_INET, socket.AF_INET6):
    comparison = Comparison(0, 4, family, 0)
    if lib.seccomp_rule_add_array(context, 0x00050000 | errno.EPERM, syscall, 1, ctypes.byref(comparison)) != 0:
        raise RuntimeError("network guard setup failed")
if lib.seccomp_load(context) != 0:
    raise RuntimeError("network guard load failed")
lib.seccomp_release(context)
for family in (socket.AF_INET, socket.AF_INET6):
    try:
        socket.socket(family, socket.SOCK_STREAM)
    except PermissionError:
        continue
    raise RuntimeError("network guard verification failed")
os.environ.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none",
                  ROCR_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  WANDB_MODE="offline", PYTHONDONTWRITEBYTECODE="1",
                  OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
os.execvp(sys.argv[1], sys.argv[1:])
