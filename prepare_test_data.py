import numpy as np

# Dimensions définies dans notre graphe
B, IN_F, OUT_F = 1, 16, 8

# 1. Entrée (x) : 1x16 float32
x = np.random.randn(B, IN_F).astype(np.float32)
x.tofile("input_x.raw")

# 2. Cible (y) : 1x8 float32
y = np.random.randn(B, OUT_F).astype(np.float32)
y.tofile("target_y.raw")

# 3. Poids latents de départ : 8x16 float32
w = np.random.randn(OUT_F, IN_F).astype(np.float32)
w.tofile("weights_latent_in.raw")

print("Fichiers .raw générés avec succès !")
