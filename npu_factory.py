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
    return res.returncode == 0

def main():
    # 1. Build
    run_cmd(f'"{PYTHON_VENV}" generate_transformer_graph.py', "Génération ONNX")
    
    converter_path = f"{BIN_PYTHON}/qnn-onnx-converter"
    run_cmd(f'powershell.exe -ExecutionPolicy Bypass -Command ". \'{SDK_ROOT}/bin/envsetup.ps1\'; & \'{SYS_PYTHON}\' \'{converter_path}\' -i nano_transformer_train.onnx --output_path qnn_transformer_model.cpp"', "Conversion QNN")
    
    src_files = ["qnn_transformer_model.cpp", "build_manual/QnnModel.cpp", "build_manual/QnnModelPal.cpp", "build_manual/QnnWrapperUtils.cpp"]
    with open("build_manual/transformer_engine.cpp", "w", encoding='utf-8') as f_out:
        for f_name in src_files:
            if os.path.exists(f_name):
                with open(f_name, "r", encoding='utf-8', errors='ignore') as f_in:
                    f_out.write(f_in.read().replace("QNN_API", 'extern "C" __declspec(dllexport)') + "\n")
    
    run_cmd(f'"{CLANG_EXE}" -shared -o test_npu/qnn_transformer_model.dll build_manual/transformer_engine.cpp build_manual/binary_mock.cpp -I"{SDK_ROOT}/include/QNN" -I"." -D_USRDLL -D_WINDLL -lkernel32 -luser32 -ladvapi32', "Compilation DLL")
    
    shutil.copy(f"{BIN_NATIVE}/qnn-context-binary-generator.exe", "test_npu/")
    for dll in ["QnnHtp.dll", "QnnSystem.dll", "QnnCpu.dll"]: shutil.copy(f"{LIB_DIR}/{dll}", "test_npu/")
    
    run_cmd(f'cd test_npu && .\\qnn-context-binary-generator.exe --model qnn_transformer_model.dll --backend QnnHtp.dll --binary_file qnn_transformer_htp.bin --output_dir output', "Binaire NPU")

    # 2. Training Loop (Hybride)
    D, SEQ = 16, 4
    w_enc = np.random.randn(D, D).astype(np.float32) * 0.1
    w_dec = np.random.randn(D, D).astype(np.float32) * 0.1
    src = np.zeros((SEQ, D), dtype=np.float32)
    src[0, :] = 1.0 
    src.tofile("test_npu/src.raw")
    src.tofile("test_npu/tgt.raw")

    print(f"\nDébut de l'entraînement Hybride (NPU=Gradients, CPU=Update)...")
    LR = 0.1
    for i in range(20):
        w_enc.tofile("test_npu/w_enc.raw")
        w_dec.tofile("test_npu/w_dec.raw")
        
        step_cmd = f'cd test_npu && .\\qnn-net-run.exe --retrieve_context output/qnn_transformer_htp.bin.bin --backend QnnHtp.dll --input_list input_list_transformer.txt --output_dir iter_out'
        subprocess.run(step_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        res_path = "test_npu/iter_out/Result_0"
        if os.path.exists(res_path):
            # Récupérer les GRADIENTS générés par le NPU
            grad_enc = np.fromfile(os.path.join(res_path, "grad_w_enc.raw"), dtype=np.float32).reshape(D, D)
            grad_dec = np.fromfile(os.path.join(res_path, "grad_w_dec.raw"), dtype=np.float32).reshape(D, D)
            pred = np.fromfile(os.path.join(res_path, "prediction.raw"), dtype=np.float32).reshape(SEQ, D)
            
            # UPDATE SUR CPU
            w_enc = w_enc - LR * grad_enc
            w_dec = w_dec - LR * grad_dec
            
            loss = np.mean((pred - src)**2)
            print(f"Step {i:02d} | Loss: {loss:.6f}")
            shutil.rmtree("test_npu/iter_out")
        else:
            print(f"Step {i:02d} | Erreur NPU")

if __name__ == "__main__":
    main()
