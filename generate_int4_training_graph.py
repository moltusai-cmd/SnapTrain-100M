import torch
import torch.nn as nn

class INT4TrainingStep(nn.Module):
    def __init__(self, in_features, out_features, lr=0.01):
        super().__init__()
        self.lr = lr
        
    def quantize_int4(self, x):
        # Fake quantization (simulates INT4 math but keeps floating point type for ONNX export)
        # INT4 range: -8 to +7, we use -7 to 7 for symmetry
        scale = x.abs().max() / 7.0
        scale = torch.clamp(scale, min=1e-5) # avoid division by zero
        x_q = torch.round(x / scale)
        x_q = torch.clamp(x_q, -7.0, 7.0)
        return x_q * scale, scale

    def quantize_ternary(self, w):
        # BitNet b1.58 style: -1, 0, 1
        scale = w.abs().mean()
        scale = torch.clamp(scale, min=1e-5)
        w_q = torch.round(w / scale)
        w_q = torch.clamp(w_q, -1.0, 1.0)
        return w_q * scale, scale

    def forward(self, x, y, w_latent):
        # --- FORWARD PASS ---
        # 1. Quantize weights to Ternary (-1, 0, 1)
        w_ternary, _ = self.quantize_ternary(w_latent)
        
        # 2. Quantize input to INT4
        x_int4, _ = self.quantize_int4(x)
        
        # 3. Linear projection: x @ w^T
        # Shape: x is (B, in_f), w is (out_f, in_f)
        out = torch.matmul(x_int4, w_ternary.t())
        
        # --- LOSS CALCULATION (MSE) ---
        # Loss = 0.5 * (out - y)^2
        # dLoss/dout = (out - y)
        grad_out = out - y
        
        # --- BACKWARD PASS (INT4) ---
        # 4. Quantize gradients to INT4
        grad_out_int4, _ = self.quantize_int4(grad_out)
        
        # 5. Calculate weight gradients
        # grad_w = grad_out^T @ x
        # Shape: grad_out is (B, out_f), x_int4 is (B, in_f) -> grad_w is (out_f, in_f)
        grad_w = torch.matmul(grad_out_int4.t(), x_int4)
        
        # --- WEIGHT UPDATE ---
        # 6. Update latent weights
        w_latent_new = w_latent - self.lr * grad_w
        
        return w_latent_new, out

# Model dimensions (Toy model)
B, IN_F, OUT_F = 1, 16, 8

model = INT4TrainingStep(in_features=IN_F, out_features=OUT_F)

# Dummy inputs
x_dummy = torch.randn(B, IN_F)
y_dummy = torch.randn(B, OUT_F)
w_dummy = torch.randn(OUT_F, IN_F)

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    (x_dummy, y_dummy, w_dummy),
    "bitnet_int4_training_step.onnx",
    input_names=["input_x", "target_y", "weights_latent_in"],
    output_names=["weights_latent_out", "prediction"],
    opset_version=18
)
print("Saved bitnet_int4_training_step.onnx")
