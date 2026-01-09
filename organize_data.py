# organize_data.py - 整理數據集

import os
import shutil

print("=" * 60)
print("開始整理數據集...")
print("=" * 60)

source_folder = 'dataset'

if not os.path.exists(source_folder):
    print(f"❌ 找不到 {source_folder} 文件夾！")
    exit(1)

print(f"✓ 找到數據集: {source_folder}")

train_source = os.path.join(source_folder, 'Train')
test_source = os.path.join(source_folder, 'Test')

if not os.path.exists(train_source):
    print("❌ 找不到 Train 文件夾！")
    exit(1)

print(f"✓ 找到訓練數據: {train_source}")

if os.path.exists(test_source):
    print(f"✓ 找到測試數據: {test_source}")

# 複製訓練數據
print("\n正在複製訓練數據...")
train_folders = os.listdir(train_source)

for folder in train_folders:
    folder_path = os.path.join(train_source, folder)
    if os.path.isdir(folder_path):
        # 判斷是 fresh 還是 rotten
        if folder.lower().startswith('fresh'):
            dest = 'data/train/fresh'
        elif folder.lower().startswith('rotten'):
            dest = 'data/train/stale'
        else:
            continue  # 跳過不認識的文件夾
        
        # 複製所有圖片
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(files) > 0:
            print(f"  複製 {len(files)} 張圖片從 {folder} 到 {dest}...")
            
            for file in files:
                src_file = os.path.join(folder_path, file)
                # 為了避免文件名衝突，加上文件夾名稱前綴
                new_filename = f"{folder}_{file}"
                dest_file = os.path.join(dest, new_filename)
                shutil.copy2(src_file, dest_file)

# 複製測試數據
if os.path.exists(test_source):
    print("\n正在複製測試數據...")
    test_folders = os.listdir(test_source)
    
    for folder in test_folders:
        folder_path = os.path.join(test_source, folder)
        if os.path.isdir(folder_path):
            # 判斷是 fresh 還是 rotten
            if folder.lower().startswith('fresh'):
                dest = 'data/test/fresh'
            elif folder.lower().startswith('rotten'):
                dest = 'data/test/stale'
            else:
                continue
            
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if len(files) > 0:
                print(f"  複製 {len(files)} 張圖片從 {folder} 到 {dest}...")
                
                for file in files:
                    src_file = os.path.join(folder_path, file)
                    new_filename = f"{folder}_{file}"
                    dest_file = os.path.join(dest, new_filename)
                    shutil.copy2(src_file, dest_file)

# 統計結果
print("\n" + "=" * 60)
print("數據整理完成！")
print("=" * 60)

train_fresh = len(os.listdir('data/train/fresh'))
train_stale = len(os.listdir('data/train/stale'))
test_fresh = len(os.listdir('data/test/fresh'))
test_stale = len(os.listdir('data/test/stale'))

print(f"\n訓練集:")
print(f"  新鮮: {train_fresh} 張")
print(f"  腐爛: {train_stale} 張")
print(f"  總計: {train_fresh + train_stale} 張")

print(f"\n測試集:")
print(f"  新鮮: {test_fresh} 張")
print(f"  腐爛: {test_stale} 張")
print(f"  總計: {test_fresh + test_stale} 張")

print(f"\n總樣本數: {train_fresh + train_stale + test_fresh + test_stale} 張")
print("\n✅ 可以開始訓練模型了！")
