import subprocess
import sys
   
packages = [
       'tensorflow',
       'numpy',
       'pandas',
       'matplotlib',
       'pillow',
       'streamlit',
       'scikit-learn'
   ]
   
for package in packages:
       print(f"正在安裝 {package}...")
       subprocess.check_call([sys.executable, "-m", "pip", "install", package])
   
print("✅ 全部安裝完成！")