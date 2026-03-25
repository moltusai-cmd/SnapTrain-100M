import torch
import torch.nn as nn
import torch.nn.functional as F

class HelloNPUModel(nn.Module):
    def __init__(self, d_model=16):
        super().__init__()
        self.d_model = d_model

    def forward(self, src, tgt, w_qkv, w_ffn1, w_ffn2):
        # --- FORWARD ---
        qkv = torch.matmul(src, w_qkv.t())
        attn = torch.matmul(qkv, qkv.t()) / 4.0 # sqrt(d_model)
        attn_probs = F.softmax(attn, dim=-1)
        context = torch.matmul(attn_probs, qkv)
        
        ffn_h = F.relu(torch.matmul(context, w_ffn1.t()))
        prediction = torch.matmul(ffn_h, w_ffn2.t())
        
        # --- GRADIENTS ---
        loss_grad = (prediction - tgt)
        grad_w_ffn2 = torch.matmul(loss_grad.t(), ffn_h)
        # grad_w_ffn1 must be (HIDDEN, D)
        # ffn_h = relu(context @ w_ffn1.t)
        # d_loss/d_w_ffn1 = (d_loss/d_ffn_h * d_ffn_h/d_net) @ context
        grad_ffn_h = torch.matmul(loss_grad, w_ffn2)
        grad_ffn_h[ffn_h <= 0] = 0 # ReLU derivative
        grad_w_ffn1 = torch.matmul(grad_ffn_h.t(), context)
        
        grad_w_qkv = torch.matmul(loss_grad.t(), src) # Very simplified qkv grad
        
        return prediction, grad_w_qkv, grad_w_ffn1, grad_w_ffn2

# Dimensions HELLO WORLD (SEQ=12, D=16)
SEQ, D = 12, 16
model = HelloNPUModel(d_model=D)

# Dummy inputs for export
src = torch.randn(SEQ, D)
tgt = torch.randn(SEQ, D)
w_qkv = torch.randn(D, D)
w_ffn1 = torch.randn(64, D) # Hidden size = 64
w_ffn2 = torch.randn(D, 64)

print("Exporting Hello World Transformer to ONNX...")
torch.onnx.export(
    model, (src, tgt, w_qkv, w_ffn1, w_ffn2),
    "hello_train.onnx",
    input_names=["src", "tgt", "w_qkv", "w_ffn1", "w_ffn2"],
    output_names=["prediction", "grad_qkv", "grad_ffn1", "grad_ffn2"],
    opset_version=17
)
print("Graphe 'Hello World' généré.")
