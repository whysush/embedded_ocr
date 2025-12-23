from PIL import Image
import numpy as np

TARGET_H = 32
TARGET_W = 128

def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")

    w, h = img.size
    scale = TARGET_H / h
    new_w = int(w * scale)

    import random
    new_w = int(new_w * random.uniform(0.9, 1.1))
    img = img.resize((max(1, new_w), TARGET_H), Image.BILINEAR)


    canvas = Image.new("L", (TARGET_W, TARGET_H), color=0)
    paste_w = min(new_w, TARGET_W)
    canvas.paste(img.crop((0, 0, paste_w, TARGET_H)), (0, 0))

    arr = np.array(canvas).astype(np.float32) / 255.0
    return arr        # (32, 128)
