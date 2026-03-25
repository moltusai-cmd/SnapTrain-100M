import numpy as np

# Dimensions (SEQ=4, D=16)
SEQ, D = 4, 16

# 1. src & tgt (Inputs)
np.random.randn(SEQ, D).astype(np.float32).tofile("src.raw")
np.random.randn(SEQ, D).astype(np.float32).tofile("tgt.raw")

# 2. w_enc & w_dec (Latent Weights)
np.random.randn(D, D).astype(np.float32).tofile("w_enc.raw")
np.random.randn(D, D).astype(np.float32).tofile("w_dec.raw")

print("Données Nano-Transformer générées !")
