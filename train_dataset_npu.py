import os
import subprocess
import numpy as np
import shutil

# --- CONFIGURATION ---
SDK_ROOT = r"C:\Users\ncouf\bitnet\qairt_sdk\qairt\2.26.2.240911"
CLANG_EXE = r"C:\Program Files\LLVM\bin\clang++.exe"
PYTHON_VENV = r"C:\Users\ncouf\bitnet\qnn_env\Scripts\python.exe"
BIN_DIR = os.path.join(SDK_ROOT, "bin", "aarch64-windows-msvc")
LIB_DIR = os.path.join(SDK_ROOT, "lib", "aarch64-windows-msvc")

def run_cmd(cmd, desc):
    print(f"==> {desc}...")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        return False
    return True

def main():
    # --- 1. DATASET DE TEST (4 MOTIFS) ---
    D, SEQ = 16, 4
    dataset = []
    for i in range(4):
        p = np.zeros((SEQ, D), dtype=np.float32)
        p[i, :] = 1.0  # Une ligne de 1 différente pour chaque motif
        dataset.append(p)
    
    print(f"Dataset de {len(dataset)} motifs prêt.")

    # --- 2. INITIALISATION ---
    w_enc = np.random.randn(D, D).astype(np.float32) * 0.1
    w_dec = np.random.randn(D, D).astype(np.float32) * 0.1
    
    # On s'assure que le dossier test_npu est prêt
    if not os.path.exists("test_npu/output"): os.makedirs("test_npu/output", exist_ok=True)

    print(f"\nDébut de l'entraînement sur NPU (Tâche: Mémorisation)...")
    
    epochs = 10
    for epoch in range(epochs):
        epoch_loss = 0
        for idx, data in enumerate(dataset):
            # Sauvegarder les données et poids pour cette étape
            data.tofile("test_npu/src.raw")
            data.tofile("test_npu/tgt.raw") # On utilise src comme cible
            w_enc.tofile("test_npu/w_enc.raw")
            w_dec.tofile("test_npu/w_dec.raw")
            
            # Exécution sur le NPU
            step_cmd = f'cd test_npu && .\\qnn-net-run.exe --retrieve_context output/qnn_transformer_htp.bin.bin --backend QnnHtp.dll --input_list input_list_transformer.txt --output_dir iter_out'
            subprocess.run(step_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            res_path = "test_npu/iter_out/Result_0"
            if os.path.exists(res_path):
                w_enc = np.fromfile(os.path.join(res_path, "w_enc_new.raw"), dtype=np.float32).reshape(D, D)
                w_dec = np.fromfile(os.path.join(res_path, "w_dec_new.raw"), dtype=np.float32).reshape(D, D)
                pred = np.fromfile(os.path.join(res_path, "prediction.raw"), dtype=np.float32).reshape(SEQ, D)
                
                loss = np.mean((pred - data)**2)
                epoch_loss += loss
                shutil.rmtree("test_npu/iter_out")
            else:
                print("Erreur NPU")
        
        print(f"Epoch {epoch:02d} | Moyenne Loss: {epoch_loss/len(dataset):.6f}")

if __name__ == "__main__":
    main()
