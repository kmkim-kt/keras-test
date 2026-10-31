import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

TRAIN_PATH = './examples/T_/bank-marketing-classification/train.csv'
TEST_PATH = './examples/T_/bank-marketing-classification/test_x.csv'
MODEL_SAVE_PATH = './01012345678_1.h5'
CSV_SAVE_PATH = './01012345678_1.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

x = train_df.drop(columns=['y'])
y = train_df['y'].astype(str)
x_test = test_df[x.columns].copy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print("클래스:", label_encoder.classes_)

x_train, x_val, y_train, y_val = train_test_split(
    x,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=1
)

# 수치형/범주형 구분
num_cols = x.select_dtypes(include=["number"]).columns.tolist()
cat_cols = x.select_dtypes(exclude=["number"]).columns.tolist()
print("학습 결측치:\n", x_train.isnull().sum())
print("테스트 결측치:\n", x_test.isnull().sum())
print("수치형 분포:\n", x_train[num_cols].describe() if num_cols else "수치형 컬럼 없음")

# 결측치 처리: 학습 데이터의 중앙값/최빈값을 모든 데이터에 동일하게 적용
if num_cols:
    median_value = x_train[num_cols].median().fillna(0)
    x_train[num_cols] = x_train[num_cols].fillna(median_value)
    x_val[num_cols] = x_val[num_cols].fillna(median_value)
    x_test[num_cols] = x_test[num_cols].fillna(median_value)

if cat_cols:
    mode_value = x_train[cat_cols].mode().iloc[0] if not x_train[cat_cols].mode().empty else pd.Series(index=cat_cols, dtype=object)
    mode_value = mode_value.fillna("missing")
    x_train[cat_cols] = x_train[cat_cols].fillna(mode_value)
    x_val[cat_cols] = x_val[cat_cols].fillna(mode_value)
    x_test[cat_cols] = x_test[cat_cols].fillna(mode_value)

# 이상치 확인: 학습 데이터의 IQR 기준과 데이터별 이상치 개수
if num_cols:
    q1 = x_train[num_cols].quantile(0.25)
    q3 = x_train[num_cols].quantile(0.75)
    iqr = q3 - q1
    # IQR이 0인 컬럼은 자동 처리에서 제외
    outlier_cols = iqr[iqr > 0].index.tolist()
    lower = q1[outlier_cols] - 1.5 * iqr[outlier_cols]
    upper = q3[outlier_cols] + 1.5 * iqr[outlier_cols]
    median = x_train[outlier_cols].median()

    outlier_info = pd.DataFrame({
        "하한": lower,
        "상한": upper,
        "대체 중앙값": median,
        "학습 이상치 수": ((x_train[outlier_cols] < lower) | (x_train[outlier_cols] > upper)).sum(),
        "검증 이상치 수": ((x_val[outlier_cols] < lower) | (x_val[outlier_cols] > upper)).sum(),
        "테스트 이상치 수": ((x_test[outlier_cols] < lower) | (x_test[outlier_cols] > upper)).sum(),
    })
    print("IQR 이상치 확인 (IQR이 0인 컬럼 제외)")
    display(outlier_info)

# 아래부터 별도 셀로 실행하면 위 표를 확인한 뒤 처리 여부를 결정할 수 있음
handle_outliers = False
if handle_outliers and num_cols:
    x_train[outlier_cols] = x_train[outlier_cols].mask((x_train[outlier_cols] < lower) | (x_train[outlier_cols] > upper), median, axis=1)
    x_val[outlier_cols] = x_val[outlier_cols].mask((x_val[outlier_cols] < lower) | (x_val[outlier_cols] > upper), median, axis=1)
    x_test[outlier_cols] = x_test[outlier_cols].mask((x_test[outlier_cols] < lower) | (x_test[outlier_cols] > upper), median, axis=1)

# 전처리: 학습 데이터로만 기준을 학습
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

x_train = preprocessor.fit_transform(x_train)
x_val = preprocessor.transform(x_val)
x_test = preprocessor.transform(x_test)

# 원-핫 인코딩 결과가 희소 행렬이면 Keras 입력을 위해 변환
if hasattr(x_train, "toarray"):
    x_train = x_train.toarray()
    x_val = x_val.toarray()
    x_test = x_test.toarray()

# 이진/다중 분류 공통: 클래스 수만큼 softmax 출력
model = Sequential(
    [
        Dense(256, activation="relu", input_shape=(x_train.shape[1],)),
        Dropout(0.1),
        Dense(128, activation="relu"),
        Dropout(0.1),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    batch_size=64,
    callbacks=[EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
    verbose=1
)

val_probs = model.predict(x_val)
val_classes = np.argmax(val_probs, axis=1)
print("validation accuracy:", accuracy_score(y_val, val_classes))
# 다중/이진 분류의 macro 평균 (문제 지시에 따라 macro, weighted 등 변경)
# print("validation F1:", f1_score(y_val, val_classes, average="macro", zero_division=0))
# print("validation precision:", precision_score(y_val, val_classes, average="macro", zero_division=0))
# print("validation recall:", recall_score(y_val, val_classes, average="macro", zero_division=0))
# 이진 ROC-AUC: "yes"를 문제에서 지정한 양성 라벨로 변경
# positive_class = label_encoder.transform(["yes"])[0]
# print("validation AUC:", roc_auc_score(y_val == positive_class, val_probs[:, positive_class]))

test_probs = model.predict(x_test)
test_classes = np.argmax(test_probs, axis=1)
test_labels = label_encoder.inverse_transform(test_classes)

result_df = test_df.copy()
result_df["y"] = test_labels
result_df.to_csv(CSV_SAVE_PATH, index=False, encoding="utf-8-sig")
model.save(MODEL_SAVE_PATH)
