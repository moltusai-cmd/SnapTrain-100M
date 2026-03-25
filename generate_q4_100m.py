import torch
import torch.nn as nn
import torch.nn.functional as F

class Q4Transformer100M(nn.Module):
    def __init__(self, d_model=1024, ffn_dim=4096, n_layers=8):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
    def quantize_int4(self, x):
        # Fake Quantization pour le QAT (Quantization-Aware Training) simulé en INT4
        scale = x.abs().max() / 7.0
        scale = torch.clamp(scale, min=1e-5)
        x_q = torch.round(x / scale)
        x_q = torch.clamp(x_q, -7.0, 7.0)
        return x_q * scale

    def forward(self, x, target, lr, *weights):
        # weights = [w_qkv_0, w_ffn1_0, w_ffn2_0, w_qkv_1, ...]
        hidden = x
        
        # Historique pour la backward pass manuelle
        ctx_list = []
        
        # --- FORWARD PASS ---
        for i in range(self.n_layers):
            w_qkv = weights[i*3]
            w_ffn1 = weights[i*3+1]
            w_ffn2 = weights[i*3+2]
            
            # Quantification des poids à la volée (Q4)
            w_qkv_q = self.quantize_int4(w_qkv)
            w_ffn1_q = self.quantize_int4(w_ffn1)
            w_ffn2_q = self.quantize_int4(w_ffn2)
            
            # Attention (Simplifiée pour le NPU)
            qkv = torch.matmul(hidden, w_qkv_q.t())
            attn = torch.matmul(qkv, qkv.t()) / (self.d_model ** 0.5)
            attn_probs = F.softmax(attn, dim=-1)
            context = torch.matmul(attn_probs, qkv)
            
            # FFN
            ffn_h = F.relu(torch.matmul(context, w_ffn1_q.t()))
            out = torch.matmul(ffn_h, w_ffn2_q.t())
            
            ctx_list.append((hidden, context, ffn_h, w_qkv_q, w_ffn1_q, w_ffn2_q))
            hidden = out
            
        prediction = hidden
        
        # --- BACKWARD PASS ---
        loss_grad = prediction - target
        new_weights = []
        grad_hidden = loss_grad
        
        # Rétropropagation à travers les 8 couches
        for i in reversed(range(self.n_layers)):
            h_in, context, ffn_h, w_qkv_q, w_ffn1_q, w_ffn2_q = ctx_list[i]
            
            # Gradients FFN2
            grad_w_ffn2 = torch.matmul(grad_hidden.t(), ffn_h)
            grad_ffn_h = torch.matmul(grad_hidden, w_ffn2_q)
            grad_ffn_h[ffn_h <= 0] = 0 # Dérivée ReLU
            
            # Gradients FFN1
            grad_w_ffn1 = torch.matmul(grad_ffn_h.t(), context)
            grad_context = torch.matmul(grad_ffn_h, w_ffn1_q)
            
            # Gradients QKV (Backward Attention simplifié)
            grad_w_qkv = torch.matmul(grad_context.t(), h_in)
            grad_hidden = torch.matmul(grad_context, w_qkv_q) 
            
            # Récupération des poids latents FP32
            w_qkv = weights[i*3]
            w_ffn1 = weights[i*3+1]
            w_ffn2 = weights[i*3+2]
            
            # Mise à jour avec le LR
            new_weights.insert(0, w_ffn2 - lr * grad_w_ffn2)
            new_weights.insert(0, w_ffn1 - lr * grad_w_ffn1)
            new_weights.insert(0, w_qkv - lr * grad_w_qkv)
            
        return tuple([prediction] + new_weights)

D = 1024
FFN = 4096
LAYERS = 8 # 8 couches de ~12M = ~96M paramètres au total
SEQ = 64

model = Q4Transformer100M(d_model=D, ffn_dim=FFN, n_layers=LAYERS)

src = torch.randn(SEQ, D)
tgt = torch.randn(SEQ, D)
lr = torch.tensor([0.001])

weights = []
for _ in range(LAYERS):
    weights.append(torch.randn(D, D) * 0.02)
    weights.append(torch.randn(FFN, D) * 0.02)
    weights.append(torch.randn(D, FFN) * 0.02)

input_names = ["src", "tgt", "lr"]
for i in range(LAYERS):
    input_names.extend([f"w_qkv_{i}", f"w_ffn1_{i}", f"w_ffn2_{i}"])

output_names = ["prediction"]
for i in range(LAYERS):
    output_names.extend([f"new_w_qkv_{i}", f"new_w_ffn1_{i}", f"new_w_ffn2_{i}"])

print(f"Exporting ~100M Parameters Q4 Transformer to ONNX...")
torch.onnx.export(
    model, (src, tgt, lr, *weights),
    "q4_100m_train.onnx",
    input_names=input_names,
    output_names=output_names,
    opset_version=18
)
print("100M Q4 ONNX Graph Exported successfully.")
