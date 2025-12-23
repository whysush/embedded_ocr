from label_encoder import encode_label, decode_label, BLANK_INDEX

word = "you123"

encoded = encode_label(word)
decoded = decode_label(encoded)

print("Word:", word)
print("Encoded:", encoded)
print("Decoded:", decoded)
print("Blank index:", BLANK_INDEX)
