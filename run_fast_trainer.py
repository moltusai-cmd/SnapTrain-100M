import ctypes
import os

BASE_DIR = r"C:\Users\ncouf\bitnet\reproduction_package"
# Important pour Windows on ARM: enregistrer le dossier des DLLs
os.add_dll_directory(BASE_DIR)

DLL_PATH = os.path.join(BASE_DIR, "npu_bridge_arm64.dll")
HTP_DLL = os.path.join(BASE_DIR, "QnnHtp.dll")
SYS_DLL = os.path.join(BASE_DIR, "QnnSystem.dll")
BIN_FILE = os.path.join(BASE_DIR, "bitnet_choc.bin")

print(f"==> Loading Bridge DLL (ARM64): {DLL_PATH}")
bridge = ctypes.CDLL(DLL_PATH)

# Signature: void run_npu_training(const char* htp, const char* sys, const char* bin, int steps)
bridge.run_npu_training.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
bridge.run_npu_training.restype = None

print("==> Launching NPU In-Memory Training (1000 steps)...")
bridge.run_npu_training(
    HTP_DLL.encode('utf-8'),
    SYS_DLL.encode('utf-8'),
    BIN_FILE.encode('utf-8'),
    1000
)
print("==> Done.")
