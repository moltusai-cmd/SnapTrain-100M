import numpy as np
import re

# Un petit échantillon de texte type Wiki
text = """
The Snapdragon is a suite of system on a chip (SoC) semiconductor products for mobile devices designed and marketed by Qualcomm Technologies Inc. 
The Snapdragon's central processing unit (CPU) uses the ARM architecture. 
A single chip may contain multiple CPU cores, a graphics processing unit (GPU), a wireless modem, and other software and hardware to support a smartphone's global positioning system (GPS), camera, gesture recognition and video. 
Snapdragon semiconductors are embedded in devices of various systems, including Android, Windows Phone and netbooks.
""" * 100 # On le multiplie pour avoir un petit dataset

# Tokenizer simple
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

def encode(s): return [char_to_ix[c] for c in s]
def decode(l): return ''.join([ix_to_char[i] for i in l])

data = np.array(encode(text), dtype=np.int32)
print(f"Vocab size: {vocab_size}, Data length: {len(data)}")

# Sauvegarde pour le script d'entraînement
np.save("vocab_meta.npy", {'chars': chars, 'char_to_ix': char_to_ix, 'ix_to_char': ix_to_char})
data.tofile("wikitext_simple.bin")
