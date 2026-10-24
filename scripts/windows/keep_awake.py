"""Keep Windows from sleeping while a flag file exists (per-process request; no system settings changed)."""
import ctypes
import os
import sys
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
flag = sys.argv[1]
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
try:
    while os.path.exists(flag):
        time.sleep(30)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
