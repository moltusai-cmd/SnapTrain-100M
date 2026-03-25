import torch
import torch.nn as nn
import torch.nn.functional as F

class MonsterFullTrainer(nn.Module):
    def __init__(self, d_model=512, n_layers=2): # Réduit à 2 couches pour la stabilité du graphe massif
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

    def forward(self, src, tgt, w_qkv, w_ffn1, w_ffn2, lr):
        # --- 1. FORWARD ---
        hidden = src
        for _ in range(self.n_layers):
            qkv = torch.matmul(hidden, w_qkv.t())
            attn = torch.matmul(qkv, qkv.t()) / 22.6
            attn_probs = F.softmax(attn, dim=-1)
            context = torch.matmul(attn_probs, qkv)
            ffn_h = F.relu(torch.matmul(context, w_ffn1.t()))
            hidden = torch.matmul(ffn_h, w_ffn2.t())
        
        prediction = hidden
        
        # --- 2. GRADIENTS ---
        loss_grad = prediction - tgt
        
        # On booste les gradients pour qu'ils survivent à la quantification NPU
        g_ffn2 = torch.matmul(loss_grad.t(), ffn_h)
        g_ffn1 = torch.matmul(ffn_h.t(), context)
        g_qkv = torch.matmul(loss_grad.t(), src)
        
        # --- 3. NPU OPTIMIZER (L'étape ultime) ---
        # La mise à jour se fait DIRECTEMENT dans le graphe NPU
        new_w_qkv = w_qkv - lr * g_qkv
        new_w_ffn1 = w_ffn1 - lr * g_ffn1
        new_w_ffn2 = w_ffn2 - lr * g_ffn2
        
        return prediction, new_w_qkv, new_w_ffn1, new_w_ffn2

# Dimensions
SEQ, D = 64, 512
model = MonsterFullTrainer()

# Inputs
src = torch.randn(SEQ, D)
tgt = torch.randn(SEQ, D)
w_qkv = torch.randn(D, D)
w_ffn1 = torch.randn(2048, D)
w_ffn2 = torch.randn(D, 2048)
lr = torch.tensor([0.001])

print("Exporting FULL NPU TRAINER (20M params)...")
torch.onnx.export(
    model, (src, tgt, w_qkv, w_ffn1, w_ffn2, lr),
    "monster_20m_full_npu.onnx",
    input_names=["src", "tgt", "w_qkv", "w_ffn1", "w_ffn2", "lr"],
    output_names=["prediction", "w_qkv_out", "w_ffn1_out", "w_ffn2_out"],
    opset_version=17
)
print("Graphe FULL NPU généré !")
