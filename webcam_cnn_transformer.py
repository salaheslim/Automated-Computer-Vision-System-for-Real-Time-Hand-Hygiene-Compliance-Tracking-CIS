# Line 1 — change model name
MODEL_NAME = "CNN + Transformer"

# Line 2 — change model file
model = tf.keras.models.load_model('cnn_transformer_model.h5')

# Line 3 — change colour to amber
ACCENT = (30, 140, 210)   # replace PURPLE with ACCENT throughout
