import subprocess

subprocess.run([
    "python", "-m", "tf2onnx.convert",
    "--onnx", "ocr_crnn.onnx",
    "--output", "ocr_tf",
    "--opset", "18"
], check=True)

print("Converted to TensorFlow SavedModel: ocr_tf/")
