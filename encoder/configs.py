VOCAB_SIZE = 20_000
MAX_LENGTH = 128
QK_SIZE = 16
EMBED_SIZE = 128
N_HEADS = 3
N_LAYERS = 2
NUM_CLASSES = 3
DROPOUT = 0.1
BATCH_SIZE = 64
EPOCHS = 8

config = dict(
    epochs=8,
    classes=3,
    vocab_size=20000,
    batch_size=64,
    learning_rate=0.005,
    max_length=128,
    qk_size=16,
    embed_size=128,
    n_heads=3,
    n_layers=2,
    dropout=0.1
)
