import numpy as np

# Alphabet minimal pour HELLO WORLD
# On mappe chaque caractère à un index
alphabet = " HELLOWRD"
char_to_idx = {c: i for i, c in enumerate(alphabet)}
idx_to_char = {i: c for i, c in enumerate(alphabet)}

text = "HELLO WORLD " * 100 # On répète pour avoir un peu de volume
tokens = np.array([char_to_idx[c] for c in text], dtype=np.int32)

tokens.tofile("hello_data.bin")

print(f"Dataset 'Hello World' généré : {len(tokens)} tokens.")
print(f"Alphabet : '{alphabet}' (Taille: {len(alphabet)})")
