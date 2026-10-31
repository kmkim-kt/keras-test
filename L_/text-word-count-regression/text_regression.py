import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

TRAIN_PATH = './examples/L_/text-word-count-regression/train.csv'
TEST_PATH = './examples/L_/text-word-count-regression/test_x.csv'
MODEL_SAVE_PATH = './01012345678_2.h5'
CSV_SAVE_PATH = './01012345678_2.csv'

MAX_WORDS = 20000
MAX_LEN = 80

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

x = train_df["문자내용"].astype(str)
y = train_df["단어수"].astype(float)
x_test = test_df["문자내용"].astype(str)

x_train, x_val, y_train, y_val = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)

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
        Dense(1)
    ]
)

model.compile(
    optimizer=Adam(0.001),
    loss="mse",
    metrics=[RootMeanSquaredError(name="rmse")],
    # metrics=["mae"],  # 평균 절대 오차
    # metrics=["mse"],  # 평균 제곱 오차
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1)
]

model.fit(
    x_train_pad,
    y_train,
    validation_data=(x_val_pad, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# metrics를 변경하면 아래 변수명과 출력도 함께 변경
val_loss, val_rmse = model.evaluate(x_val_pad, y_val, verbose=0)
print("validation RMSE:", val_rmse)

pred_values = model.predict(x_test_pad).reshape(-1)

result_df = test_df.copy()
result_df["단어수"] = pred_values

result_df.to_csv(CSV_SAVE_PATH, index=False, encoding="utf-8-sig")
model.save(MODEL_SAVE_PATH)
