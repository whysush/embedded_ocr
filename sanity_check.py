import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt

mat = sio.loadmat("data/IIIT5K/trainCharBound.mat")
train_cb = mat["trainCharBound"]

sample = train_cb[0][0]

chars = sample["chars"][0]
word = "".join([c[0] for c in chars])

img = Image.open("data/IIIT5K/train/1009_2.png")

plt.imshow(img)
plt.title(word)
plt.axis("off")
plt.show()
