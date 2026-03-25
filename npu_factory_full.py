import os
import subprocess
import numpy as np
import shutil
import time

# --- CONFIGURATION ---
SDK_ROOT = "C:/Users/ncouf/bitnet/qairt_sdk/qairt/2.26.2.240911"
CLANG_EXE = "C:/Program Files/LLVM/bin/clang++.exe"
SYS_PYTHON = "C:/Users/ncouf/AppData/Local/Programs/Python/Python310/python.exe"
PYTHON_VENV = "C:/Users/ncouf/bitnet/qnn_env/Scripts/python.exe"
BIN_PYTHON = f"{SDK_ROOT}/bin/arm64x-windows-msvc"
BIN_NATIVE = f"{SDK_ROOT}/bin/aarch64-windows-msvc"
LIB_DIR = f"{SDK_ROOT}/lib/aarch64-windows-msvc"

def run_cmd(cmd, desc):
    print(f"==> {desc}...")
    my_env = os.environ.copy()
    my_env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=my_env, encoding='utf-8', errors='replace')
    if res.returncode != 0:
        print(f"ERREUR : {res.stderr}")
        return False
    return True

def build_full():
    # 1. ONNX
    if not run_cmd(f'"{PYTHON_VENV}" generate_monster_full.py', "Génération ONNX (Full)"): return False
    
    # 2. QNN C++
    converter_path = f"{BIN_PYTHON}/qnn-onnx-converter"
    env_cmd = f'powershell.exe -ExecutionPolicy Bypass -Command ". \'{SDK_ROOT}/bin/envsetup.ps1\'; & \'{SYS_PYTHON}\' \'{converter_path}\' -i monster_20m_full_npu.onnx --output_path qnn_monster_full.cpp"'
    if not run_cmd(env_cmd, "Conversion QNN"): return False
    
    # 3. Fusion C++
    print("==> Fusion C++ (Full)...")
    src_files = ["qnn_monster_full.cpp", "build_manual/QnnModel.cpp", "build_manual/QnnModelPal.cpp", "build_manual/QnnWrapperUtils.cpp"]
    with open("build_manual/full_engine.cpp", "w", encoding='utf-8') as f_out:
        for f_name in src_files:
            if os.path.exists(f_name):
                with open(f_name, "r", encoding='utf-8', errors='ignore') as f_in:
                    f_out.write(f_in.read().replace("QNN_API", 'extern "C" __declspec(dllexport)') + "\n")
    
    # 4. DLL
    compile_cmd = f'"{CLANG_EXE}" -shared -o test_npu/qnn_monster_full.dll build_manual/full_engine.cpp build_manual/binary_mock.cpp -I"{SDK_ROOT}/include/QNN" -I"." -D_USRDLL -D_WINDLL -lkernel32 -luser32 -ladvapi32'
    if not run_cmd(compile_cmd, "Compilation DLL"): return False
    
    # 5. Serialization
    print("==> Sérialisation NPU (Full)...")
    shutil.copy(f"{BIN_NATIVE}/qnn-context-binary-generator.exe", "test_npu/")
    serialize_cmd = f'cd test_npu && .\\qnn-context-binary-generator.exe --model qnn_monster_full.dll --backend QnnHtp.dll --binary_file monster_full_htp.bin --output_dir output'
    if not run_cmd(serialize_cmd, "Binaire NPU"): return False
    return True

def train_full():
    D, SEQ = 512, 64
    VOCAB_SIZE = 8001
    
    data = np.fromfile("wikitext_simple.bin", dtype=np.int32)
    embedding_table = np.random.randn(VOCAB_SIZE, D).astype(np.float32) * 0.1
    
    # Initial weights
    w_qkv = (np.random.randn(D, D) * 0.02).astype(np.float32)
    w_ffn1 = (np.random.randn(2048, D) * 0.02).astype(np.float32)
    w_ffn2 = (np.random.randn(D, 2048) * 0.02).astype(np.float32)
    lr = np.array([0.01], dtype=np.float32)

    with open("test_npu/input_list_full.txt", "w") as f:
        f.write("src:=src.raw tgt:=tgt.raw w_qkv:=w_qkv.raw w_ffn1:=w_ffn1.raw w_ffn2:=w_ffn2.raw lr:=lr.raw")

    print("\nLancement de l'entraînement FULL NPU...")
    for step in range(20):
        idx = np.random.randint(0, len(data) - SEQ - 1)
        src = embedding_table[data[idx:idx+SEQ]]
        tgt = embedding_table[data[idx+1:idx+SEQ+1]]
        
        src.tofile("test_npu/src.raw")
        tgt.tofile("test_npu/tgt.raw")
        w_qkv.tofile("test_npu/w_qkv.raw")
        w_ffn1.tofile("test_npu/w_ffn1.raw")
        w_ffn2.tofile("test_npu/w_ffn2.raw")
        lr.tofile("test_npu/lr.raw")
        
        # Step
        subprocess.run(f'cd test_npu && .\\qnn-net-run.exe --retrieve_context output/monster_full_htp.bin.bin --backend QnnHtp.dll --input_list input_list_full.txt --output_dir full_out', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        res_path = "test_npu/full_out/Result_0"
        if os.path.exists(res_path):
            old_w_qkv = w_qkv.copy()
            old_w_ffn1 = w_ffn1.copy()
            old_w_ffn2 = w_ffn2.copy()
            
            # ON RÉCUPÈRE LES NOUVEAUX POIDS CALCUÉS PAR LE NPU
            w_qkv = np.fromfile(os.path.join(res_path, "w_qkv_out.raw"), dtype=np.float32).reshape(D, D)
            w_ffn1 = np.fromfile(os.path.join(res_path, "w_ffn1_out.raw"), dtype=np.float32).reshape(2048, D)
            w_ffn2 = np.fromfile(os.path.join(res_path, "w_ffn2_out.raw"), dtype=np.float32).reshape(D, 2048)
            pred = np.fromfile(os.path.join(res_path, "prediction.raw"), dtype=np.float32).reshape(SEQ, D)
            
            gn_qkv = (old_w_qkv - w_qkv) / lr[0]
            gn_ffn1 = (old_w_ffn1 - w_ffn1) / lr[0]
            gn_ffn2 = (old_w_ffn2 - w_ffn2) / lr[0]
            gn = np.sqrt(np.sum(gn_qkv**2) + np.sum(gn_ffn1**2) + np.sum(gn_ffn2**2))
            
            loss = np.mean((pred - tgt)**2)
            print(f"Step {step:02d} | Loss: {loss:.6f} | LR: {lr[0]} | GN: {gn:.6f}")
            shutil.rmtree("test_npu/full_out")
        else:
            print(f"Step {step:02d} | Erreur")

if __name__ == "__main__":
    if build_full():
        train_full()
