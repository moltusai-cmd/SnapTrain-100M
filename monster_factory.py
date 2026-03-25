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

def build_monster():
    if not run_cmd(f'"{PYTHON_VENV}" generate_monster_graph.py', "Génération ONNX"): return False
    
    converter_path = f"{BIN_PYTHON}/qnn-onnx-converter"
    if not run_cmd(f'powershell.exe -ExecutionPolicy Bypass -Command ". \'{SDK_ROOT}/bin/envsetup.ps1\'; & \'{SYS_PYTHON}\' \'{converter_path}\' -i monster_20m_train.onnx --output_path qnn_monster_model.cpp"', "Conversion QNN"): return False
    
    print("==> Fusion C++...")
    src_files = ["qnn_monster_model.cpp", "build_manual/QnnModel.cpp", "build_manual/QnnModelPal.cpp", "build_manual/QnnWrapperUtils.cpp"]
    with open("build_manual/monster_engine.cpp", "w", encoding='utf-8') as f_out:
        for f_name in src_files:
            if os.path.exists(f_name):
                with open(f_name, "r", encoding='utf-8', errors='ignore') as f_in:
                    f_out.write(f_in.read().replace("QNN_API", 'extern "C" __declspec(dllexport)') + "\n")
    
    if not run_cmd(f'"{CLANG_EXE}" -shared -o test_npu/qnn_monster_model.dll build_manual/monster_engine.cpp build_manual/binary_mock.cpp -I"{SDK_ROOT}/include/QNN" -I"." -D_USRDLL -D_WINDLL -lkernel32 -luser32 -ladvapi32', "Compilation DLL"): return False
    
    shutil.copy(f"{BIN_NATIVE}/qnn-context-binary-generator.exe", "test_npu/")
    for dll in ["QnnHtp.dll", "QnnSystem.dll", "QnnCpu.dll"]: shutil.copy(f"{LIB_DIR}/{dll}", "test_npu/")
    
    if not run_cmd(f'cd test_npu && .\\qnn-context-binary-generator.exe --model qnn_monster_model.dll --backend QnnHtp.dll --binary_file qnn_monster_htp.bin --output_dir output', "Binaire NPU"): return False
    return True

def train_monster():
    D, SEQ = 512, 64
    VOCAB_SIZE = 8001
    
    # 1. Chargement des données
    data = np.fromfile("wikitext_simple.bin", dtype=np.int32)
    print(f"Entraînement sur {len(data)} tokens Wikitext...")

    # 2. Initialisation des POIDS (20M)
    embedding_table = np.random.randn(VOCAB_SIZE, D).astype(np.float32) * 0.1
    w_qkv = np.random.randn(D, D).astype(np.float32) * 0.02
    w_ffn1 = np.random.randn(2048, D).astype(np.float32) * 0.02
    w_ffn2 = np.random.randn(D, 2048).astype(np.float32) * 0.02

    LR = 0.001
    
    # input_list pour Monster
    with open("test_npu/input_list_monster.txt", "w") as f:
        f.write("src:=src.raw tgt:=tgt.raw w_qkv:=w_qkv.raw w_ffn1:=w_ffn1.raw w_ffn2:=w_ffn2.raw")

    print("\nLancement de l'entraînement MONSTER sur NPU Snapdragon...")
    for step in range(50):
        # Tirage d'un batch
        idx = np.random.randint(0, len(data) - SEQ - 1)
        x_tokens = data[idx:idx+SEQ]
        y_tokens = data[idx+1:idx+SEQ+1]
        
        # CPU : Embedding (Tokens -> Vectors)
        src = embedding_table[x_tokens]
        tgt = embedding_table[y_tokens]
        
        # Sauvegarde pour NPU
        src.tofile("test_npu/src.raw")
        tgt.tofile("test_npu/tgt.raw")
        w_qkv.tofile("test_npu/w_qkv.raw")
        w_ffn1.tofile("test_npu/w_ffn1.raw")
        w_ffn2.tofile("test_npu/w_ffn2.raw")
        
        # NPU : Calcul des Gradients
        subprocess.run(f'cd test_npu && .\\qnn-net-run.exe --retrieve_context output/qnn_monster_htp.bin.bin --backend QnnHtp.dll --input_list input_list_monster.txt --output_dir monster_out', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        res_path = "test_npu/monster_out/Result_0"
        if os.path.exists(res_path):
            g_qkv = np.fromfile(os.path.join(res_path, "grad_qkv.raw"), dtype=np.float32).reshape(D, D)
            g_ffn1 = np.fromfile(os.path.join(res_path, "grad_ffn1.raw"), dtype=np.float32).reshape(2048, D)
            g_ffn2 = np.fromfile(os.path.join(res_path, "grad_ffn2.raw"), dtype=np.float32).reshape(D, 2048)
            pred = np.fromfile(os.path.join(res_path, "prediction.raw"), dtype=np.float32).reshape(SEQ, D)
            
            # CPU : Update
            w_qkv -= LR * g_qkv
            w_ffn1 -= LR * g_ffn1
            w_ffn2 -= LR * g_ffn2
            
            gn = np.sqrt(np.sum(g_qkv**2) + np.sum(g_ffn1**2) + np.sum(g_ffn2**2))
            loss = np.mean((pred - tgt)**2)
            print(f"Step {step:02d} | Loss: {loss:.6f} | LR: {LR} | GN: {gn:.6f}")
            shutil.rmtree("test_npu/monster_out")
        else:
            print(f"Step {step:02d} | Erreur NPU")

if __name__ == "__main__":
    if build_monster():
        train_monster()
