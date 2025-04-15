import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

# GPU 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU 사용 가능: {gpus[0].name}")
else:
    print("GPU를 사용할 수 없습니다. CPU로 학습됩니다.")

# 1. 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 전처리
x_train = x_train / 255.0  # 정규화
x_test = x_test / 255.0
y_train = to_categorical(y_train)  # 원-핫 인코딩
y_test = to_categorical(y_test)

# 3. 모델 구성
model = Sequential([
    Flatten(input_shape=(28, 28)),     # 28x28 이미지를 1차원으로 변환
    Dense(128, activation='relu'),     # 은닉층
    Dense(10, activation='softmax')    # 출력층: 10개 클래스 (0~9)
])

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
model.fit(x_train, y_train, epochs=5, batch_size=64)

# 6. 모델 평가
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\n테스트 정확도: {accuracy * 100:.2f}%")
