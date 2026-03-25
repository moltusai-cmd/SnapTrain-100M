import ctypes
import os

BASE_DIR = r"C:\Users\ncouf\bitnet\reproduction_package"
# On enregistre le dossier du SDK x64 pour QnnHtp.dll
SDK_X64 = r"C:\Users\ncouf\bitnet\qairt_sdk\qairt\2.26.2.240911\lib\x86_64-windows-msvc"
os.add_dll_directory(SDK_X64)
os.add_dll_directory(BASE_DIR)

DLL_PATH = os.path.join(BASE_DIR, "npu_bridge_nosys.dll")
HTP_DLL = os.path.join(SDK_X64, "QnnHtp.dll")
BIN_FILE = os.path.join(BASE_DIR, "bitnet_choc.bin")

print(f"==> Loading No-Sys Bridge DLL: {DLL_PATH}")
bridge = ctypes.CDLL(DLL_PATH)

# Signature: void run_npu_training_no_sys(const char* htp, const char* bin, int steps)
bridge.run_npu_training_no_sys.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
bridge.run_npu_training_no_sys.restype = None

print("==> Launching NPU In-Memory Training (No-Sys, 1000 steps)...")
bridge.run_npu_training_no_sys(
    HTP_DLL.encode('utf-8'),
    BIN_FILE.encode('utf-8'),
    1000
)
print("==> Done.")
