import ctypes
import os
import sys

DLL_DIR = r"C:\Users\ncouf\bitnet\qairt_sdk\qairt\2.26.2.240911\lib\aarch64-windows-msvc"
if os.path.exists(DLL_DIR):
    os.add_dll_directory(DLL_DIR)
    os.environ["PATH"] = DLL_DIR + os.pathsep + os.environ["PATH"]

HTP_DLL = "QnnHtp.dll"
SYS_DLL = "QnnSystem.dll"

print(f"==> Python NPU Bridge Loader 🐉")

try:
    # On utilise LoadLibraryEx via ctypes pour être sûr
    lib_sys = ctypes.CDLL(os.path.join(DLL_DIR, SYS_DLL))
    lib_htp = ctypes.CDLL(os.path.join(DLL_DIR, HTP_DLL))
    print("DLLs loaded successfully!")
    
    # Test interface
    lib_htp.QnnInterface_getProviders.restype = ctypes.c_uint32
    print("QnnInterface_getProviders is ready.")
    
except Exception as e:
    print(f"ERROR: {e}")
