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

def build_hello():
    if not run_cmd(f'"{PYTHON_VENV}" generate_hello_graph.py', "Génération ONNX"): return False
    
    converter_path = f"{BIN_PYTHON}/qnn-onnx-converter"
    if not run_cmd(f'powershell.exe -ExecutionPolicy Bypass -Command ". \'{SDK_ROOT}/bin/envsetup.ps1\'; & \'{SYS_PYTHON}\' \'{converter_path}\' -i hello_train.onnx --output_path qnn_hello_model.cpp"', "Conversion QNN"): return False
    
    src_files = ["qnn_hello_model.cpp", "build_manual/QnnModel.cpp", "build_manual/QnnModelPal.cpp", "build_manual/QnnWrapperUtils.cpp"]
    with open("build_manual/hello_engine.cpp", "w", encoding='utf-8') as f_out:
        for f_name in src_files:
            if os.path.exists(f_name):
                with open(f_name, "r", encoding='utf-8', errors='ignore') as f_in:
                    f_out.write(f_in.read().replace("QNN_API", 'extern "C" __declspec(dllexport)') + "\n")
    
    if not run_cmd(f'"{CLANG_EXE}" -shared -o test_npu/qnn_hello_model.dll build_manual/hello_engine.cpp build_manual/binary_mock.cpp -I"{SDK_ROOT}/include/QNN" -I"." -D_USRDLL -D_WINDLL -lkernel32 -luser32 -ladvapi32', "Compilation DLL"): return False
    
    shutil.copy(f"{BIN_NATIVE}/qnn-context-binary-generator.exe", "test_npu/")
    for dll in ["QnnHtp.dll", "QnnSystem.dll", "QnnCpu.dll"]: shutil.copy(f"{LIB_DIR}/{dll}", "test_npu/")
    
    if not run_cmd(f'cd test_npu && .\\qnn-context-binary-generator.exe --model qnn_hello_model.dll --backend QnnHtp.dll --binary_file hello_htp.bin --output_dir output', "Binaire NPU"): return False
    return True

def train_hello():
    D, SEQ = 16, 12
    HIDDEN = 64
    alphabet = " HELLOWRD"
    VOCAB_SIZE = len(alphabet)
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    idx_to_char = {i: c for i, c in enumerate(alphabet)}

    # Initialisation
    embedding_table = np.random.randn(VOCAB_SIZE, D).astype(np.float32) * 0.1
    w_qkv = np.random.randn(D, D).astype(np.float32) * 0.1
    w_ffn1 = np.random.randn(HIDDEN, D).astype(np.float32) * 0.1
    w_ffn2 = np.random.randn(D, HIDDEN).astype(np.float32) * 0.1
    LR = 0.05

    data = np.fromfile("hello_data.bin", dtype=np.int32)
    
    with open("test_npu/input_list_hello.txt", "w") as f:
        f.write("src:=src.raw tgt:=tgt.raw w_qkv:=w_qkv.raw w_ffn1:=w_ffn1.raw w_ffn2:=w_ffn2.raw")

    print("\nEntraînement HELLO WORLD sur NPU...")
    for step in range(100):
        # On prend une séquence fixe "HELLO WORLD "
        idx = 0
        x_tokens = data[idx:idx+SEQ]
        y_tokens = data[idx+1:idx+SEQ+1] # Décalé de 1
        
        src = embedding_table[x_tokens]
        tgt = embedding_table[y_tokens]
        
        src.tofile("test_npu/src.raw")
        tgt.tofile("test_npu/tgt.raw")
        w_qkv.tofile("test_npu/w_qkv.raw")
        w_ffn1.tofile("test_npu/w_ffn1.raw")
        w_ffn2.tofile("test_npu/w_ffn2.raw")
        
        subprocess.run(f'cd test_npu && .\\qnn-net-run.exe --retrieve_context output/hello_htp.bin.bin --backend QnnHtp.dll --input_list input_list_hello.txt --output_dir hello_out', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        res_path = "test_npu/hello_out/Result_0"
        if os.path.exists(res_path):
            g_qkv = np.fromfile(os.path.join(res_path, "grad_qkv.raw"), dtype=np.float32).reshape(D, D)
            g_ffn1 = np.fromfile(os.path.join(res_path, "grad_ffn1.raw"), dtype=np.float32).reshape(HIDDEN, D)
            g_ffn2 = np.fromfile(os.path.join(res_path, "grad_ffn2.raw"), dtype=np.float32).reshape(D, HIDDEN)
            pred = np.fromfile(os.path.join(res_path, "prediction.raw"), dtype=np.float32).reshape(SEQ, D)
            
            # Update
            w_qkv -= LR * g_qkv
            w_ffn1 -= LR * g_ffn1
            w_ffn2 -= LR * g_ffn2
            
            if step % 20 == 0:
                loss = np.mean((pred - tgt)**2)
                print(f"Step {step:03d} | Loss: {loss:.6f}")
            shutil.rmtree("test_npu/hello_out")

    # --- INFERENCE TEST ---
    print("\n--- TEST D'INFÉRENCE ---")
    input_str = "HELL"
    print(f"Input: '{input_str}'")
    
    # 1. Encoding
    current_tokens = [char_to_idx[c] for c in input_str]
    # Padding pour atteindre SEQ=12
    while len(current_tokens) < SEQ: current_tokens.append(0)
    
    src_test = embedding_table[current_tokens]
    src_test.tofile("test_npu/src.raw")
    # tgt bidon pour l'inférence
    np.zeros((SEQ, D), dtype=np.float32).tofile("test_npu/tgt.raw")
    
    subprocess.run(f'cd test_npu && .\\qnn-net-run.exe --retrieve_context output/hello_htp.bin.bin --backend QnnHtp.dll --input_list input_list_hello.txt --output_dir hello_inf', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    res_path = "test_npu/hello_inf/Result_0"
    if os.path.exists(res_path):
        pred = np.fromfile(os.path.join(res_path, "prediction.raw"), dtype=np.float32).reshape(SEQ, D)
        
        # 2. Decoding (Recherche du vecteur embedding le plus proche)
        output_text = ""
        for i in range(len(input_str), SEQ):
            vec = pred[i-1] # Le modèle prédit le caractère suivant celui à l'index i-1
            dists = np.linalg.norm(embedding_table - vec, axis=1)
            next_char_idx = np.argmin(dists)
            output_text += idx_to_char[next_char_idx]
        
        print(f"Prediction: '{input_str}{output_text}'")
    else:
        print("Erreur d'inférence.")

if __name__ == "__main__":
    if build_hello():
        train_hello()
