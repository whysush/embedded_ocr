import scipy.io as sio
import matplotlib.pyplot as plt
from preprocess import preprocess_image

# Load labels
mat = sio.loadmat("data/IIIT5K/trainCharBound.mat")
train_cb = mat["trainCharBound"]

# Take first sample
sample = train_cb[0][0]

img_name = sample["ImgName"][0]
chars = sample["chars"][0]
word = "".join([c[0] for c in chars])

img_path = f"data/IIIT5K/{img_name}"

processed = preprocess_image(img_path)

plt.imshow(processed[0], cmap="gray")
plt.title(word)
plt.axis("off")
plt.show()
