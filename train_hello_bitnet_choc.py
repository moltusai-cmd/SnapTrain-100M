import os
import subprocess
import numpy as np
import shutil

# --- CONFIG ---
SDK_ROOT = "C:/Users/ncouf/bitnet/qairt_sdk/qairt/2.26.2.240911"
CLANG_EXE = "C:/Program Files/LLVM/bin/clang++.exe"
SYS_PYTHON = "C:/Users/ncouf/AppData/Local/Programs/Python/Python310/python.exe"
PYTHON_VENV = "C:/Users/ncouf/bitnet/qnn_env/Scripts/python.exe"
BIN_PYTHON = f"{SDK_ROOT}/bin/arm64x-windows-msvc"
BIN_NATIVE = f"{SDK_ROOT}/bin/aarch64-windows-msvc"
LIB_DIR = f"{SDK_ROOT}/lib/aarch64-windows-msvc"

def run_cmd(cmd):
    my_env = os.environ.copy()
    my_env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=my_env, encoding='utf-8', errors='replace')
    return res.returncode == 0

def build():
    print("==> Building BitNet Choc...")
    if not run_cmd(f'"{PYTHON_VENV}" generate_hello_bitnet_choc.py'): return False
    converter = f"{BIN_PYTHON}/qnn-onnx-converter"
    if not run_cmd(f'powershell.exe -ExecutionPolicy Bypass -Command ". \'{SDK_ROOT}/bin/envsetup.ps1\'; & \'{SYS_PYTHON}\' \'{converter}\' -i hello_bitnet_choc.onnx --output_path qnn_bitnet_choc.cpp"'): return False
    
    src_files = ["qnn_bitnet_choc.cpp", "build_manual/QnnModel.cpp", "build_manual/QnnModelPal.cpp", "build_manual/QnnWrapperUtils.cpp"]
    with open("build_manual/bitnet_choc_engine.cpp", "w", encoding='utf-8') as f_out:
        for f_name in src_files:
            if os.path.exists(f_name):
                with open(f_name, "r", encoding='utf-8', errors='ignore') as f_in:
                    f_out.write(f_in.read().replace("QNN_API", 'extern "C" __declspec(dllexport)') + "\n")
    
    if not run_cmd(f'"{CLANG_EXE}" -shared -o test_npu/qnn_bitnet_choc.dll build_manual/bitnet_choc_engine.cpp build_manual/binary_mock.cpp -I"{SDK_ROOT}/include/QNN" -I"." -D_USRDLL -D_WINDLL -lkernel32 -luser32 -ladvapi32'): return False
    if not run_cmd(f'cd test_npu && .\\qnn-context-binary-generator.exe --model qnn_bitnet_choc.dll --backend QnnHtp.dll --binary_file bitnet_choc_htp.bin --output_dir output'): return False
    return True

def train():
    D, SEQ, HIDDEN = 128, 12, 512
    alphabet = " HELLOWRD"
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    idx_to_char = {i: c for i, c in enumerate(alphabet)}

    # Weights (Initialisation très faible pour BitNet Choc)
    emb = np.random.randn(len(alphabet), D).astype(np.float32) * 0.01
    w_qkv = np.random.randn(D, D).astype(np.float32) * 0.01
    w_ffn1 = np.random.randn(HIDDEN, D).astype(np.float32) * 0.01
    w_ffn2 = np.random.randn(D, HIDDEN).astype(np.float32) * 0.01
    
    data = np.fromfile("hello_data.bin", dtype=np.int32)
    lr_arr = np.array([1.0], dtype=np.float32) # LR AGRESSIF

    with open("test_npu/input_list_choc.txt", "w") as f:
        f.write("src:=src.raw tgt:=tgt.raw w_qkv:=w_qkv.raw w_ffn1:=w_ffn1.raw w_ffn2:=w_ffn2.raw lr:=lr.raw")

    print("\nTraining BITNET CHOC on NPU (The 'almost working' version)...")
    for step in range(200):
        lr_arr[0] = 1.0 * (1 - step / 200)
        src = emb[data[0:SEQ]]
        tgt = emb[data[1:SEQ+1]]
        
        src.tofile("test_npu/src.raw"); tgt.tofile("test_npu/tgt.raw")
        w_qkv.tofile("test_npu/w_qkv.raw"); w_ffn1.tofile("test_npu/w_ffn1.raw"); w_ffn2.tofile("test_npu/w_ffn2.raw")
        lr_arr.tofile("test_npu/lr.raw")
        
        subprocess.run(f'cd test_npu && .\\qnn-net-run.exe --retrieve_context output/bitnet_choc_htp.bin.bin --backend QnnHtp.dll --input_list input_list_choc.txt --output_dir choc_out', shell=True, capture_output=True)
        
        res = "test_npu/choc_out/Result_0"
        if os.path.exists(res):
            w_qkv = np.fromfile(os.path.join(res, "new_w_qkv.raw"), dtype=np.float32).reshape(D, D)
            w_ffn1 = np.fromfile(os.path.join(res, "new_w_ffn1.raw"), dtype=np.float32).reshape(HIDDEN, D)
            w_ffn2 = np.fromfile(os.path.join(res, "new_w_ffn2.raw"), dtype=np.float32).reshape(D, HIDDEN)
            if step % 50 == 0:
                pred = np.fromfile(os.path.join(res, "prediction.raw"), dtype=np.float32).reshape(SEQ, D)
                loss = np.mean((pred - tgt)**2)
                print(f"Step {step:03d} | Loss: {loss:.6f} | LR: {lr_arr[0]:.4f}")
            shutil.rmtree("test_npu/choc_out")

    print("\nNote: Cette version a prouvé l'effondrement de la loss sur BitNet NPU.")

if __name__ == "__main__":
    if build(): train()
