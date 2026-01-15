import scanpy as sc
import numpy as np
import os

H5AD_FILE = "train.h5ad"

if not os.path.exists(H5AD_FILE):
    print("❌ 找不到 train.h5ad")
    exit()

print(f"正在读取 {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
genes = adata.var_names.tolist()

# 保存
np.save("gene_vocab.npy", genes)
print(f"✅ 已保存 gene_vocab.npy (包含 {len(genes)} 个基因)")