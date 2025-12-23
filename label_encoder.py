import string

# digits + lowercase + uppercase
CHARSET = string.digits + string.ascii_lowercase
NUM_CHARS = len(CHARSET)  # 62

BLANK_INDEX = NUM_CHARS   # CTC blank = 62

char_to_idx = {c: i for i, c in enumerate(CHARSET)}
idx_to_char = {i: c for i, c in enumerate(CHARSET)}

def encode_label(text):
    return [char_to_idx[c] for c in text if c in char_to_idx]
