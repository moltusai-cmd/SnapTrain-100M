import ctypes
import os

dll_path = os.path.abspath("qnn_training_model.dll")
print(f"Tentative de chargement de : {dll_path}")

try:
    # On ajoute le dossier actuel au chemin de recherche des DLL (Python 3.8+)
    os.add_dll_directory(os.getcwd())
    os.add_dll_directory(r"C:\Users\ncouf\bitnet\qairt_sdk\qairt\2.26.2.240911\lib\arm64x-windows-msvc")
    
    lib = ctypes.CDLL(dll_path)
    print("DLL chargée avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement : {e}")
