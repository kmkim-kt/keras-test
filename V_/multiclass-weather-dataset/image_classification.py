import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import F1Score  # macro F1 사용 시

TRAIN_DIR = './examples/V_/multiclass-weather-dataset/train'
TEST_DIR = './examples/V_/multiclass-weather-dataset/test'
TEST_PATH = './examples/V_/multiclass-weather-dataset/test_x.csv'
MODEL_SAVE_PATH = './01012345678_2.h5'
CSV_SAVE_PATH = './01012345678_2.csv'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 학습 데이터에만 가벼운 증강 적용
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    horizontal_flip=True,
    # rotation_range=10,
    # zoom_range=0.1,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
)
val_generator = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
)
print('클래스:', train_generator.class_indices)

# 테스트 CSV의 파일명 순서대로 이미지를 읽음
result_df = pd.read_csv(TEST_PATH)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=result_df,
    directory=TEST_DIR,
    x_col='파일명',
    y_col=None,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
)

# 이진/다중 분류 모두 클래스 수만큼 softmax 출력
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(len(train_generator.class_indices), activation='softmax')(x)
model = Model(base_model.input, output)

model.compile(
    optimizer=Adam(0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    # metrics=['accuracy', F1Score(average='macro', name='f1_score')],
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    # EarlyStopping(
    #                 monitor="val_accuracy",
    #                 mode="max",
    #                 patience=5,
    #                 restore_best_weights=True,
    #                 verbose=1
    #             ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1),
]


model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks,
    verbose=1,
)

# softmax 출력은 클래스별 확률이므로 argmax로 클래스 선택
pred_probs = model.predict(test_generator)
pred_classes = np.argmax(pred_probs, axis=1)
index_to_class = {index: name for name, index in train_generator.class_indices.items()}
pred_labels = [index_to_class[index] for index in pred_classes]

result_df['라벨'] = pred_labels
print(result_df.head())
print(result_df['라벨'].value_counts())

result_df.to_csv(CSV_SAVE_PATH, index=False, encoding='utf-8-sig')
model.save(MODEL_SAVE_PATH)
