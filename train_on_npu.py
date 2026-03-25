import numpy as np
import subprocess
import os
import shutil

# Config
ITERATIONS = 50
D = 16
SEQ = 4

# Initialisation des poids
w_enc = np.random.randn(D, D).astype(np.float32)
w_dec = np.random.randn(D, D).astype(np.float32)

# Données fixes (On veut que le transformer apprenne à reconstruire src)
src = np.random.randn(SEQ, D).astype(np.float32)
tgt = np.random.randn(SEQ, D).astype(np.float32)

src.tofile("test_npu/src.raw")
tgt.tofile("test_npu/tgt.raw")

print(f"Début de l'entraînement sur NPU ({ITERATIONS} itérations)...")

losses = []

for i in range(ITERATIONS):
    # 1. Sauvegarder les poids actuels pour le NPU
    w_enc.tofile("test_npu/w_enc.raw")
    w_dec.tofile("test_npu/w_dec.raw")
    
    # 2. Lancer l'exécution sur le NPU
    # On utilise qnn-net-run.exe en mode silencieux
    cmd = [
        "test_npu/qnn-net-run.exe",
        "--retrieve_context", "test_npu/output/qnn_transformer_htp.bin.bin",
        "--backend", "test_npu/QnnHtp.dll",
        "--input_list", "test_npu/input_list_transformer.txt",
        "--output_dir", "test_npu/training_iter"
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 3. Récupérer les nouveaux poids et la prédiction
    res_path = "test_npu/training_iter/Result_0"
    pred = np.fromfile(os.path.join(res_path, "prediction.raw"), dtype=np.float32).reshape(SEQ, D)
    w_enc = np.fromfile(os.path.join(res_path, "w_enc_new.raw"), dtype=np.float32).reshape(D, D)
    w_dec = np.fromfile(os.path.join(res_path, "w_dec_new.raw"), dtype=np.float32).reshape(D, D)
    
    # 4. Calculer la Loss (MSE) pour le suivi
    loss = np.mean((pred - src)**2)
    losses.append(loss)
    
    if i % 5 == 0:
        print(f"Iteration {i:03d} | Loss: {loss:.6f}")

    # Nettoyage pour l'itération suivante
    shutil.rmtree("test_npu/training_iter")

print("\nEntraînement terminé !")
print(f"Loss initiale: {losses[0]:.6f}")
print(f"Loss finale:   {losses[-1]:.6f}")

if losses[-1] < losses[0]:
    print("SUCCÈS : Le Transformer a appris sur le NPU ! 🏆🐉")
else:
    print("STAGNATION : Ajustez le learning rate ou l'architecture.")
