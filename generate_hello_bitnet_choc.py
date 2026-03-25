import torch
import torch.nn as nn
import torch.nn.functional as F

class HelloBitNetChoc(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model

    def quantize_ternary(self, w):
        # Version "Choc" : sign(w) pur, pas de scale pour forcer le basculement initial
        return torch.sign(w)

    def forward(self, src, tgt, w_qkv, w_ffn1, w_ffn2, lr):
        # 1. Ternarisation
        w_qkv_q = self.quantize_ternary(w_qkv)
        w_ffn1_q = self.quantize_ternary(w_ffn1)
        w_ffn2_q = self.quantize_ternary(w_ffn2)
        
        # 2. Forward (Activations FP32 pour stabiliser les gradients sur NPU)
        qkv = torch.matmul(src, w_qkv_q.t())
        attn = torch.matmul(qkv, qkv.t()) / 11.31
        attn_probs = F.softmax(attn, dim=-1)
        context = torch.matmul(attn_probs, qkv)
        
        ffn_h = F.relu(torch.matmul(context, w_ffn1_q.t()))
        prediction = torch.matmul(ffn_h, w_ffn2_q.t())
        
        # 3. Gradients
        loss_grad = (prediction - tgt)
        g_ffn2 = torch.matmul(loss_grad.t(), ffn_h)
        grad_ffn_h = torch.matmul(loss_grad, w_ffn2_q)
        grad_ffn_h[ffn_h <= 0] = 0
        g_ffn1 = torch.matmul(grad_ffn_h.t(), context)
        g_qkv = torch.matmul(loss_grad.t(), src)
        
        # 4. Update
        new_w_qkv = w_qkv - lr * g_qkv
        new_w_ffn1 = w_ffn1 - lr * g_ffn1
        new_w_ffn2 = w_ffn2 - lr * g_ffn2
        
        return prediction, new_w_qkv, new_w_ffn1, new_w_ffn2

SEQ, D, HIDDEN = 12, 128, 512
model = HelloBitNetChoc(d_model=D)

# Export
src = torch.randn(SEQ, D)
tgt = torch.randn(SEQ, D)
w_qkv = torch.randn(D, D) * 0.01 
w_ffn1 = torch.randn(HIDDEN, D) * 0.01
w_ffn2 = torch.randn(D, HIDDEN) * 0.01
lr = torch.tensor([1.0])

print("Exporting BITNET CHOC...")
torch.onnx.export(
    model, (src, tgt, w_qkv, w_ffn1, w_ffn2, lr),
    "hello_bitnet_choc.onnx",
    input_names=["src", "tgt", "w_qkv", "w_ffn1", "w_ffn2", "lr"],
    output_names=["prediction", "new_w_qkv", "new_w_ffn1", "new_w_ffn2"],
    opset_version=18
)
