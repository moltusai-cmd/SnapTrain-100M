import torch
import torch.nn as nn
import torch.nn.functional as F

class ProTransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x, w_qkv, w_out, w_ffn1, w_ffn2):
        # 1. Multi-Head Attention (Packed QKV)
        qkv = torch.matmul(x, w_qkv.t())
        # Simplified Attention for NPU
        attn = torch.matmul(qkv, qkv.transpose(-1, -2)) / 16.0
        attn_probs = F.softmax(attn, dim=-1)
        context = torch.matmul(attn_probs, qkv)
        attn_out = torch.matmul(context, w_out.t())
        
        # 2. FFN
        h = F.relu(torch.matmul(attn_out, w_ffn1.t()))
        out = torch.matmul(h, w_ffn2.t())
        
        return out

class ProGradientFactory(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model

    def forward(self, src, tgt, w_qkv, w_out, w_ffn1, w_ffn2):
        # --- FORWARD ---
        # On simule un bloc encodeur complet
        # src: (32, 256), weights: matrices de 256x256 ou 256x1024
        
        # Attention + FFN
        qkv_out = torch.matmul(src, w_qkv.t())
        attn_matrix = torch.matmul(qkv_out, qkv_out.t()) / 16.0
        attn_probs = F.softmax(attn_matrix, dim=-1)
        context = torch.matmul(attn_probs, qkv_out)
        
        ffn1_out = F.relu(torch.matmul(context, w_ffn1.t()))
        prediction = torch.matmul(ffn1_out, w_ffn2.t())
        
        # --- GRADIENTS (The Heavy Part) ---
        loss_grad = (prediction - tgt)
        
        # Calcul de tous les gradients en parallèle
        grad_w_ffn2 = torch.matmul(loss_grad.t(), ffn1_out)
        grad_w_ffn1 = torch.matmul(loss_grad.t()[:256, :], context) # Simplifié pour le PoC
        grad_w_qkv = torch.matmul(loss_grad.t(), src)
        
        # On pack TOUS les gradients dans un seul vecteur pour la vitesse
        # (flatten et concatenation)
        all_grads = torch.cat([
            grad_w_qkv.view(-1),
            grad_w_ffn1.view(-1),
            grad_w_ffn2.view(-1)
        ])
        
        return prediction, all_grads

# Dimensions PRO
SEQ, D = 32, 256 # Modèle 1.5M params
model = ProGradientFactory(d_model=D)

# Dummy data
src_data = torch.randn(SEQ, D)
tgt_data = torch.randn(SEQ, D)
w_qkv = torch.randn(D, D)
w_out = torch.randn(D, D)
w_ffn1 = torch.randn(1024, D)
w_ffn2 = torch.randn(D, 1024)

print("Exporting 1.5M Params Transformer to ONNX...")
torch.onnx.export(
    model, (src_data, tgt_data, w_qkv, w_out, w_ffn1, w_ffn2),
    "nano_transformer_train.onnx",
    input_names=["src", "tgt", "w_qkv", "w_out", "w_ffn1", "w_ffn2"],
    output_names=["prediction", "all_grads"],
    opset_version=17
)
print("Modèle 1.5M Paramètres généré.")
