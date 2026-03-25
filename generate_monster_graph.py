import torch
import torch.nn as nn
import torch.nn.functional as F

class MonsterTransformer(nn.Module):
    def __init__(self, d_model=512, n_layers=4):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

    def forward(self, x, target, w_qkv, w_ffn1, w_ffn2):
        # x: (Seq, D) = (64, 512)
        # target: (64, 512)
        # w_qkv: (512, 512), w_ffn1: (2048, 512), w_ffn2: (512, 2048)
        
        hidden = x
        
        # On simule 4 couches
        for _ in range(self.n_layers):
            # 1. Attention (Self-Attention simplifiée)
            # Q, K, V sont toutes issues de w_qkv projection
            qkv = torch.matmul(hidden, w_qkv.t()) # (64, 512)
            
            # Matrice d'attention : (64, 512) @ (512, 64) -> (64, 64)
            attn = torch.matmul(qkv, qkv.t()) / 22.6 # sqrt(d_model)
            attn_probs = F.softmax(attn, dim=-1)
            
            # Contexte : (64, 64) @ (64, 512) -> (64, 512)
            context = torch.matmul(attn_probs, qkv)
            
            # 2. FFN
            # Projection 1 : (64, 512) @ (512, 2048) -> (64, 2048)
            ffn_h = F.relu(torch.matmul(context, w_ffn1.t()))
            
            # Projection 2 : (64, 2048) @ (2048, 512) -> (64, 512)
            hidden = torch.matmul(ffn_h, w_ffn2.t())
        
        prediction = hidden
        
        # --- GRADIENTS ---
        loss_grad = prediction - target
        
        # Calcul des gradients simplifiés pour le NPU
        grad_w_ffn2 = torch.matmul(loss_grad.t(), ffn_h) # (512, 2048)
        grad_w_ffn1 = torch.matmul(ffn_h.t(), context)   # (2048, 512)
        grad_w_qkv = torch.matmul(loss_grad.t(), x)      # (512, 512)
        
        return prediction, grad_w_qkv, grad_w_ffn1, grad_w_ffn2

# Dimensions MONSTER
SEQ, D = 64, 512
model = MonsterTransformer(d_model=D, n_layers=4)

# Inputs
src = torch.randn(SEQ, D)
tgt = torch.randn(SEQ, D)
w_qkv = torch.randn(D, D)
w_ffn1 = torch.randn(2048, D)
w_ffn2 = torch.randn(D, 2048)

print("Exporting Monster-20M Transformer...")
torch.onnx.export(
    model, (src, tgt, w_qkv, w_ffn1, w_ffn2),
    "monster_20m_train.onnx",
    input_names=["src", "tgt", "w_qkv", "w_ffn1", "w_ffn2"],
    output_names=["prediction", "grad_qkv", "grad_ffn1", "grad_ffn2"],
    opset_version=17
)
print("Graphe Monster-20M corrigé.")
