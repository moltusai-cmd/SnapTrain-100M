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
    print(f"\n[CMD] {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=my_env, encoding='utf-8', errors='replace')
    if res.returncode != 0:
        print("[ERROR]")
        print(res.stderr)
        return False
    return True

def build():
    # Only build if the dll doesn't exist to save time
    if os.path.exists("test_npu/qnn_q4_100m.dll") and os.path.exists("test_npu/output/q4_100m_htp.bin.bin"):
        print("==> DLL and HTP Binary already exist. Skipping build.")
        return True

    print("==> Generating 100M Q4 Graph...")
    if not run_cmd(f'"{PYTHON_VENV}" generate_q4_100m.py'): return False
    
    converter = f"{BIN_PYTHON}/qnn-onnx-converter"
    print("==> Converting to QNN C++ (this might take a few minutes for ~100M params)...")
    if not run_cmd(f'powershell.exe -ExecutionPolicy Bypass -Command ". \'{SDK_ROOT}/bin/envsetup.ps1\'; & \'{SYS_PYTHON}\' \'{converter}\' -i q4_100m_train.onnx --output_path qnn_q4_100m.cpp"'): return False
    
    print("==> Compiling DLL...")
    src_files = ["qnn_q4_100m.cpp", "build_manual/QnnModel.cpp", "build_manual/QnnModelPal.cpp", "build_manual/QnnWrapperUtils.cpp"]
    with open("build_manual/q4_100m_engine.cpp", "w", encoding='utf-8') as f_out:
        for f_name in src_files:
            if os.path.exists(f_name):
                with open(f_name, "r", encoding='utf-8', errors='ignore') as f_in:
                    f_out.write(f_in.read().replace("QNN_API", 'extern "C" __declspec(dllexport)') + "\n")
    
    if not run_cmd(f'"{CLANG_EXE}" -shared -o test_npu/qnn_q4_100m.dll build_manual/q4_100m_engine.cpp build_manual/binary_mock.cpp -I"{SDK_ROOT}/include/QNN" -I"." -D_USRDLL -D_WINDLL -lkernel32 -luser32 -ladvapi32'): return False
    
    print("==> Generating HTP Context Binary...")
    if not run_cmd(f'cd test_npu && .\\qnn-context-binary-generator.exe --model qnn_q4_100m.dll --backend QnnHtp.dll --binary_file q4_100m_htp.bin --output_dir output'): return False
    return True

def train():
    D, FFN, LAYERS, SEQ = 1024, 4096, 8, 64
    
    print("==> Loading WikiText Data...")
    if not os.path.exists("wikitext_simple.bin"):
        print("Please run prepare_wiki.py first.")
        return
    
    data = np.fromfile("wikitext_simple.bin", dtype=np.int32)
    meta = np.load("vocab_meta.npy", allow_pickle=True).item()
    vocab_size = len(meta['chars'])
    print(f"Vocab Size: {vocab_size}, Dataset length: {len(data)}")

    # Initialize Embedding Table
    embedding_table = np.random.randn(vocab_size, D).astype(np.float32) * 0.02
    
    print("==> Initializing 100M Weights in RAM...")
    weights_qkv = [np.random.randn(D, D).astype(np.float32) * 0.02 for _ in range(LAYERS)]
    weights_ffn1 = [np.random.randn(FFN, D).astype(np.float32) * 0.02 for _ in range(LAYERS)]
    weights_ffn2 = [np.random.randn(D, FFN).astype(np.float32) * 0.02 for _ in range(LAYERS)]
    
    lr_arr = np.array([0.005], dtype=np.float32)
    lr_arr.tofile("test_npu/lr.raw")
    
    input_list = "src:=src.raw tgt:=tgt.raw lr:=lr.raw "
    for i in range(LAYERS):
        input_list += f"w_qkv_{i}:=w_qkv_{i}.raw w_ffn1_{i}:=w_ffn1_{i}.raw w_ffn2_{i}:=w_ffn2_{i}.raw "
        
    with open("test_npu/input_list_100m.txt", "w") as f:
        f.write(input_list.strip())

    print("\n==> Starting 100M Q4 Training Loop on Hexagon NPU on WikiText...")
    
    epochs = 1
    steps_per_epoch = 15 # Train for a few steps to see if loss goes down
    
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            # Batch extraction
            idx = step * SEQ
            if idx + SEQ + 1 > len(data):
                idx = 0 # reset if out of bounds
                
            x_tokens = data[idx:idx+SEQ]
            y_tokens = data[idx+1:idx+SEQ+1]
            
            src = embedding_table[x_tokens]
            tgt = embedding_table[y_tokens]
            
            # Save inputs
            src.tofile("test_npu/src.raw")
            tgt.tofile("test_npu/tgt.raw")
            
            # Save current weights
            for i in range(LAYERS):
                weights_qkv[i].tofile(f"test_npu/w_qkv_{i}.raw")
                weights_ffn1[i].tofile(f"test_npu/w_ffn1_{i}.raw")
                weights_ffn2[i].tofile(f"test_npu/w_ffn2_{i}.raw")
            
            # Run NPU
            res = subprocess.run(f'cd test_npu && .\\qnn-net-run.exe --retrieve_context output/q4_100m_htp.bin.bin --backend QnnHtp.dll --input_list input_list_100m.txt --output_dir q4_100m_out', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            out_dir = "test_npu/q4_100m_out/Result_0"
            if os.path.exists(out_dir):
                # Calculate loss
                pred = np.fromfile(os.path.join(out_dir, "prediction.raw"), dtype=np.float32).reshape(SEQ, D)
                loss = np.mean((pred - tgt)**2)
                
                # Load updated weights
                for i in range(LAYERS):
                    weights_qkv[i] = np.fromfile(os.path.join(out_dir, f"new_w_qkv_{i}.raw"), dtype=np.float32).reshape(D, D)
                    weights_ffn1[i] = np.fromfile(os.path.join(out_dir, f"new_w_ffn1_{i}.raw"), dtype=np.float32).reshape(FFN, D)
                    weights_ffn2[i] = np.fromfile(os.path.join(out_dir, f"new_w_ffn2_{i}.raw"), dtype=np.float32).reshape(D, FFN)
                
                print(f"Epoch {epoch+1} | Step {step:03d} | Loss: {loss:.6f} | Weights Updated!")
                shutil.rmtree("test_npu/q4_100m_out")
            else:
                print(f"Step {step} NPU Execution failed.")
                break

if __name__ == "__main__":
    if build():
        train()
