import sys
import subprocess

cmd = [
    sys.executable, "-m", "tf2onnx.convert",
    "--onnx", "ocr_crnn.onnx",
    "--saved-model", "ocr_tf",
    "--opset", "18"
]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

print("Converted ONNX → TensorFlow SavedModel (ocr_tf)")
