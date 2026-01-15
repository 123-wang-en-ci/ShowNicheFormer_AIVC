import scanpy as sc
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ================= 配置 =================
SOURCE_FILE = "Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad"
TRAIN_FILE = "train.h5ad"
TEST_FILE = "test.h5ad"
TEST_SIZE = 0.2  # 20% 作为测试集 (严谨的科研通常用 20% 或 10%)
RANDOM_STATE = 42 # 固定随机种子，保证每次划分结果一样
# =======================================

def split_data():
    print("🚀 [数据集划分] 开始将数据集划分为训练集和测试集...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(base_dir, SOURCE_FILE)
    train_path = os.path.join(base_dir, TRAIN_FILE)
    test_path = os.path.join(base_dir, TEST_FILE)

    if not os.path.exists(source_path):
        print(f"❌ 找不到源文件: {source_path}")
        return

    # 1. 加载原始数据
    print(f"⏳ 正在读取 {SOURCE_FILE}...")
    try:
        adata = sc.read_h5ad(source_path)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return
        
    n_cells = adata.n_obs
    print(f"📄 原始数据: {n_cells} 细胞 x {adata.n_vars} 基因")

    # 2. 执行划分
    print(f"✂️ 正在按 {1-TEST_SIZE:.0%}/{TEST_SIZE:.0%} 比例划分...")
    indices = np.arange(n_cells)
    # 使用 sklearn 进行随机划分，保证分布均匀
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 3. 创建子集对象
    train_adata = adata[train_idx].copy()
    test_adata = adata[test_idx].copy()

    print(f"   - 训练集 (Train): {train_adata.n_obs} 细胞")
    print(f"   - 测试集 (Test):  {test_adata.n_obs} 细胞")

    # 4. 保存文件
    print(f"💾 正在保存 {TRAIN_FILE} ...")
    train_adata.write(train_path)
    
    print(f"💾 正在保存 {TEST_FILE} ...")
    test_adata.write(test_path)
    
    print("✅ 数据集划分完成！")
    print("👉 后续操作建议：")
    print("   1. 运行 train_imputation.py (使用 train.h5ad)")
    print("   2. 运行 evaluate_imputation.py (使用 test.h5ad)")

if __name__ == "__main__":
    split_data()