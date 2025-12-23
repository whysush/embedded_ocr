import torch
from model import OCRNet

model = OCRNet(num_classes=37)
model.load_state_dict(torch.load("ocr_crnn.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 1, 32, 128)

torch.onnx.export(
    model,
    dummy_input,
    "ocr_crnn_android.onnx",
    opset_version=13,          # ✅ IMPORTANT
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)

print("Exported ocr_crnn_android.onnx")
