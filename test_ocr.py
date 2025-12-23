import torch
import scipy.io as sio
from model import OCRNet
from preprocess import preprocess_image
from beam_decode import ctc_beam_search

model = OCRNet()
model.load_state_dict(torch.load("ocr_crnn.pth", map_location="cpu"))
model.eval()

mat = sio.loadmat("data/IIIT5K/trainCharBound.mat")
key = [k for k in mat.keys() if not k.startswith("__")][0]
samples = mat[key]

print("\nRunning OCR on TRAIN samples:\n")

for i in range(10):
    sample = samples[0][i]

    chars = sample["chars"][0]
    gt = "".join([c[0] for c in chars])

    img_name = sample["ImgName"][0]
    img_path = f"data/IIIT5K/{img_name}"

    img = preprocess_image(img_path)
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)[:, 0, :]

    pred = ctc_beam_search(logits, beam_width=5)
    print(f"GT: {gt:15s} | Pred: {pred}")

print("\nDone.")
