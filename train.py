# train.py - 訓練CNN模型

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

print("=" * 70)
print("食品品質檢測AI訓練系統")
print("=" * 70)

# 1. 設置參數
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10  # 先訓練10輪，可以改成20

print(f"\n訓練參數:")
print(f"  圖片大小: {IMG_SIZE}x{IMG_SIZE}")
print(f"  批次大小: {BATCH_SIZE}")
print(f"  訓練輪數: {EPOCHS}")

# 2. 檢查數據
print("\n檢查數據集...")
train_fresh = len(os.listdir('data/train/fresh'))
train_stale = len(os.listdir('data/train/stale'))
test_fresh = len(os.listdir('data/test/fresh'))
test_stale = len(os.listdir('data/test/stale'))

print(f"✓ 訓練集 - 新鮮: {train_fresh}, 腐爛: {train_stale}")
print(f"✓ 測試集 - 新鮮: {test_fresh}, 腐爛: {test_stale}")

# 3. 數據增強和加載
print("\n正在準備數據...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"✓ 訓練樣本數: {train_generator.samples}")
print(f"✓ 測試樣本數: {test_generator.samples}")
print(f"✓ 類別對應: {train_generator.class_indices}")

# 4. 建立模型
print("\n正在建立AI模型...")

base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✓ 模型建立完成")

# 5. 訓練模型
print("\n" + "=" * 70)
print(f"開始訓練（共{EPOCHS}輪）")
print("=" * 70 + "\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    verbose=1
)

# 6. 評估模型
print("\n" + "=" * 70)
print("訓練完成！正在評估...")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

print(f"\n📊 測試集結果:")
print(f"  準確率: {test_accuracy*100:.2f}%")
print(f"  損失: {test_loss:.4f}")

# 7. 保存模型
os.makedirs('models', exist_ok=True)
model.save('models/food_quality_detector.h5')
print(f"\n✓ 模型已保存: models/food_quality_detector.h5")

# 8. 繪製訓練曲線
print("\n正在生成訓練曲線圖...")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='訓練準確率', linewidth=2)
plt.plot(history.history['val_accuracy'], label='驗證準確率', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('準確率')
plt.title('模型準確率')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='訓練損失', linewidth=2)
plt.plot(history.history['val_loss'], label='驗證損失', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('損失')
plt.title('模型損失')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
print("✓ 訓練曲線已保存: training_history.png")

# 9. 測試幾張圖片
print("\n" + "=" * 70)
print("測試示例圖片...")
print("=" * 70)

from tensorflow.keras.preprocessing import image

# 測試新鮮食品
print("\n【測試新鮮食品】")
test_fresh_folder = 'data/test/fresh'
fresh_images = [f for f in os.listdir(test_fresh_folder) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]

for img_name in fresh_images:
    img_path = os.path.join(test_fresh_folder, img_name)
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array, verbose=0)[0][0]
    result = "🍎 新鮮" if prediction < 0.5 else "🤢 腐爛"
    confidence = (1 - prediction) if prediction < 0.5 else prediction
    
    print(f"  {img_name[:30]}: {result} (置信度: {confidence*100:.1f}%)")

# 測試腐爛食品
print("\n【測試腐爛食品】")
test_stale_folder = 'data/test/stale'
stale_images = [f for f in os.listdir(test_stale_folder) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]

for img_name in stale_images:
    img_path = os.path.join(test_stale_folder, img_name)
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array, verbose=0)[0][0]
    result = "🍎 新鮮" if prediction < 0.5 else "🤢 腐爛"
    confidence = (1 - prediction) if prediction < 0.5 else prediction
    
    print(f"  {img_name[:30]}: {result} (置信度: {confidence*100:.1f}%)")

print("\n" + "=" * 70)
print("✅ 訓練完成！")
print("=" * 70)
print("\n下一步:")
print("  1. 查看訓練曲線: training_history.png")
print("  2. 運行演示界面: streamlit run app.py")