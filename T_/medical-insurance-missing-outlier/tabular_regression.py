import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

TRAIN_PATH = './examples/T_/medical-insurance-missing-outlier/03_train.csv'
TEST_PATH = './examples/T_/medical-insurance-missing-outlier/03_test_x.csv'
MODEL_SAVE_PATH = './01012345678_3.h5'
CSV_SAVE_PATH = './01012345678_3.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

x = train_df.drop(columns=["charges"])
y = train_df["charges"]
x_test = test_df[x.columns].copy()

x_train, x_val, y_train, y_val = train_test_split(
    x,
    y,
    test_size=0.2,
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
handle_outliers = True
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

model = Sequential(
    [
        Dense(256, activation="relu", input_shape=(x_train.shape[1],)),
        Dense(128, activation="relu"),
        Dropout(0.1),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ]
)

model.compile(
    optimizer=Adam(0.001),
    loss="mse",
    metrics=[RootMeanSquaredError(name="rmse")],
    # metrics=["mae"],  # 평균 절대 오차
    # metrics=["mse"],  # 평균 제곱 오차
)

model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=300,
    batch_size=64,
    callbacks=[EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)],
    # callbacks=[
    # EarlyStopping(
    #     monitor="val_rmse",
    #     mode="min",
    #     min_delta=0.0,
    #     patience=30,
    #     restore_best_weights=True,
    #     verbose=1,
    # )],
    verbose=1
)

val_pred = model.predict(x_val).reshape(-1)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
# 다른 지표 사용 시 import 및 출력 변경
# from sklearn.metrics import mean_absolute_error
# print("validation MAE:", mean_absolute_error(y_val, val_pred))
# print("validation MSE:", mean_squared_error(y_val, val_pred))
print("validation RMSE:", rmse)

test_pred = model.predict(x_test).reshape(-1)
result_df = test_df.copy()
result_df["charges"] = test_pred
result_df.to_csv(CSV_SAVE_PATH, index=False, encoding="utf-8-sig")
model.save(MODEL_SAVE_PATH)
