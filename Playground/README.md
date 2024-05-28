🔑 **PRT(Peer Review Template)**

- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)**
    - [x] 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - [x] 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개,
    퀘스트 문제 요구조건 등을 지칭
        - [x] 해당 조건을 만족하는 부분의 코드 및 결과물을 캡쳐하여 사진으로 첨부
```py
class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compiled_loss(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(targets, predictions)
        return {m.name: m.result() for m in self.metrics}

inputs = keras.Input(shape=(28 * 28,))
features = layers.Dense(512, activation="relu")(inputs)
features = layers.Dropout(0.5)(features)
outputs = layers.Dense(10, activation="softmax")(features)
model = CustomModel(inputs, outputs)

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(train_images, train_labels, epochs=3)
```

- ~~[x]  **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**~~
    - [x] ~~모델 선정 이유: 모델이 정해져있었다.~~
    - [x]  ~~Metrics 선정 이유: 메트릭이 정해져 있었다.~~
    - [x]  ~~Loss 선정 이유: 이하 동일~~

- [x]  ~~**3. 체크리스트에 해당하는 항목들을 모두 수행하였나요? (문제 해결)**~~
    - [ ]  데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)
    - [ ]  하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)
    - [ ]  각 실험을 시각화하여 비교하였나요?
    - [ ]  모든 실험 결과가 기록되었나요?

- [x]  **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)**
    - [x]  배운 점
    - [x]  아쉬운 점
    - [x]  느낀 점
    - [x]  어려웠던 점
    

## 회고
**1. 배운 점**
 - 단순 Sequential 모델로 보다가 함수형 API, Model 서브클래싱 등으로 모델을 만드는 예시를 보며 모델의 학습 프로세스에 대해 더 잘 이해할 수 있었음.
 
**2. 아쉬운 점**
 - 아직 사직접 모델링 해본다면 여려울 것 같다는 생각이 들었음.

**3. 느낀 점**

**4. 어려웠던 점**
 - 단순 모델들만 보다가 사용자 정의 모델을 보니 코드의 복잡도가 높아 이해하기 여려웠음
 
