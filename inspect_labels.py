import scipy.io as sio

mat = sio.loadmat("data/IIIT5K/trainCharBound.mat")
train_cb = mat["trainCharBound"]

sample = train_cb[0][0]

chars = sample["chars"][0]
word = "".join([c[0] for c in chars])

print("Extracted word:", word)
