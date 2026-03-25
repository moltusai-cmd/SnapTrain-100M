# QNN Custom Op: ClusterUnpack (Near-Memory Training) 🧠⚡

Ce document détaille l'implémentation d'une opération personnalisée (Custom Op) pour le NPU Snapdragon (DSP Hexagon), visant à optimiser l'entraînement et l'inférence via le **Bit-Packing**.

---

## 🎯 Objectif : Briser le "Memory Wall"
Le but est de stocker 4 poids binaires/ternaires (2-bits chacun) dans un seul octet (`UINT8`). 
- **Stockage** : Réduit par 4 la consommation de RAM.
- **Vitesse** : Augmente par 4 la bande passante effective.
- **Calcul** : Le dépaquetage se fait "Near-Memory" directement sur le DSP.

---

## 🛠 Architecture Technique

### 1. Le Format de Grappe (Cluster)
Chaque octet `storage` est décomposé en 4 unités logiques de 2 bits :
- `w0 = val & 0x03`
- `w1 = (val >> 2) & 0x03`
- `w2 = (val >> 4) & 0x03`
- `w3 = (val >> 6) & 0x03`

### 2. Le Kernel C++ (HTP v75)
Le code est implémenté dans `cluster_op_pkg/ClusterOpPackage/src/ops/ClusterUnpack.cpp`. Il utilise les APIs de bas niveau du SDK Qualcomm pour accéder directement aux pointeurs de mémoire du DSP.

```cpp
// Extrait du Kernel
w_logical(i_b, i_h, i_w, i_d * 4 + 0) = (int8_t)((val & 0x03) < 2 ? -1 : 1);
```

---

## 🏗 Processus de Build (Windows on ARM)

La compilation d'un Op Package sur Windows est complexe car le SDK privilégie Linux. Nous avons mis au point une méthode hybride.

### Prerequisites
- **Hexagon SDK 6.5.0.0**
- **LLVM/Clang** installé dans `C:/Program Files/LLVM`.
- **GnuWin32 (make)** pour exécuter les Makefiles.

### Étapes de Compilation
Nous générons deux binaires :
1. **HTP (.so)** : Le vrai code qui tourne sur le DSP.
2. **CPU (.dll)** : Un binaire "miroir" pour Windows ARM64, nécessaire pour la phase de préparation du modèle.

Commande de build :
```powershell
# Utilise notre Makefile personnalisé pour Windows
cd cluster_op_pkg/ClusterOpPackage
make -f Makefile_win.txt htp_v75
```

---

## 🚀 Intégration dans un Modèle

### 1. Définition ONNX
Le nœud doit appartenir au domaine `bitnet` et s'appeler `ClusterUnpack`.
- Input: `w_storage` (UINT8)
- Output: `w_logical` (INT8)

### 2. Conversion QNN
Pour convertir un modèle utilisant cette op :
```powershell
qnn-onnx-converter `
  -i model.onnx `
  --op_package_config cluster_unpack_config.xml `
  --output_path qnn_model.cpp
```

### 3. Exécution sur NPU
Il faut fournir le package au runtime :
```powershell
qnn-net-run `
  --model qnn_model.dll `
  --backend QnnHtp.dll `
  --op_packages "libQnnClusterOpPackage_CPU.dll:ClusterOpPackageInterfaceProvider:CPU" `
  --op_packages "libQnnClusterOpPackage.so:ClusterOpPackageInterfaceProvider:HTP"
```

---

## ⚠️ Leçons Apprises & Troubleshooting

### FastRPC Call Retry Timeout
C'est l'erreur principale rencontrée. Elle survient quand :
- Le DSP ne parvient pas à charger le `.so` (dépendances manquantes).
- L'interface `InterfaceProvider` est mal déclarée ou corrompue.
- Le driver FastRPC est bloqué suite à un crash précédent.
- **Solution** : Redémarrer le PC pour réinitialiser le sous-système Hexagon.

### Signatures de Fonctions
Les macros `DEF_PACKAGE_OP` et `BEGIN_PKG_OP_DEFINITION` sont obligatoires. Le NPU est extrêmement strict sur la présence des stubs de log et d'initialisation, même s'ils sont vides.

---
**Développé par :** Gemini CLI x Snapdragon Master
**Status :** Kernel Compilé (v75), En attente de validation après restart.
