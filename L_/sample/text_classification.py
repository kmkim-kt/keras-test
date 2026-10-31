import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.metrics import F1Score  # macro F1 사용 시

TRIAN_PATH = './examples/L_/sample/02_train.csv'
TEST_PATH = './examples/L_/sample/02_test_x.csv'
MODEL_SAVE_PATH = './01012345678_2.h5'
CSV_SAVE_PATH = './01012345678_2.csv'

MAX_WORDS = 20000
MAX_LEN = 40

train_df = pd.read_csv(TRIAN_PATH)
test_df = pd.read_csv(TEST_PATH)

x = train_df["상품명"].astype(str)
y = train_df["카테고리"].astype(str)
x_test = test_df["상품명"].astype(str)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

x_train, x_val, y_train, y_val = train_test_split(
    x,
    y_encoded,
    stratify=y_encoded,
    test_size=0.2,
)

y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)

# 텍스트 전처리
tokenizer = Tokenizer(MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_val_seq = tokenizer.texts_to_sequences(x_val)
x_test_seq = tokenizer.texts_to_sequences(x_test)

x_train_pad = pad_sequences(x_train_seq, MAX_LEN, padding="post", truncating="post")
x_val_pad = pad_sequences(x_val_seq, MAX_LEN, padding="post", truncating="post")
x_test_pad = pad_sequences(x_test_seq, MAX_LEN, padding="post", truncating="post")

model = Sequential(
    [
        Embedding(MAX_WORDS, 128),
        Conv1D(128, 3, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ]
)

model.compile(
    optimizer=Adam(0.001),
    loss = "categorical_crossentropy",
    metrics=["accuracy"], 
    # metrics=["accuracy", "AUC"], from tensorflow.keras.metrics import F1Score
    # []"accuracy",
    # F1Score(average="macro", name="f1_score")]

)

callbacks=[
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1)
]

model.fit(
    x_train_pad,
    y_train_cat,
    validation_data=(x_val_pad, y_val_cat),
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

pred_probs = model.predict(x_test_pad)
pred_classes = np.argmax(pred_probs, axis=1)
pred_labels = label_encoder.inverse_transform(pred_classes)

result_df = test_df.copy()
result_df["카테고리"] = pred_labels

result_df.to_csv(CSV_SAVE_PATH, index=False, encoding="utf-8-sig")
model.save(MODEL_SAVE_PATH)