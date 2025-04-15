import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU 사용 가능: {gpus[0].name}")
else:
    print("GPU 사용 불가, CPU로 진행됩니다.")


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_train = x_train.astype('float32') / 255.0  # 정규화 (0~1 범위)
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)        # 원-핫 인코딩
y_test = to_categorical(y_test, 10)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=25, batch_size=64)


loss, accuracy = model.evaluate(x_test, y_test)
print(f"\n테스트 정확도: {accuracy * 100:.2f}%")
