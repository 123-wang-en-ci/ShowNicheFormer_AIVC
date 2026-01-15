import scanpy as sc
import pandas as pd
import numpy as np
import os

def analyze_tissue_segmentation_data(h5ad_file_path):
    """
    分析用于组织区域语义分割任务的数据
    """
    print("🔍 开始分析组织区域语义分割数据...")
    
    # 检查文件是否存在
    if not os.path.exists(h5ad_file_path):
        print(f"❌ 找不到文件: {h5ad_file_path}")
        return
    
    # 加载数据
    print(f"📂 正在加载数据: {h5ad_file_path}")
    adata = sc.read_h5ad(h5ad_file_path)
    print(f"✅ 成功加载 {adata.n_obs} 个细胞，{adata.n_vars} 个基因")
    
    # 1. 显示基本数据结构信息
    print("\n=== 数据基本信息 ===")
    print(f"基因元数据列名: {list(adata.var.columns)}")
    print(f"细胞元数据列名: {list(adata.obs.columns)}")
    print(f"细胞多维注释键名: {list(adata.obsm.keys()) if hasattr(adata, 'obsm') and adata.obsm else 'None'}")
    
    # 2. 分析可用于组织区域语义分割的列
    print("\n=== 组织区域语义分割相关字段分析 ===")
    segmentation_columns = [
        'slice', 'slice_id', 'tissue', 'tissue_ontology_term_id', 
        'fov', 'clust_annot','feature_types','spatial',
    ]
    
    for col in segmentation_columns:
        if col in adata.obs.columns:
            unique_values = adata.obs[col].unique()
            print(f"\n📋 [{col}] 字段信息:")
            print(f"   唯一值数量: {len(unique_values)}")
            if len(unique_values) <= 20:
                print(f"   所有唯一值: {list(unique_values)}")
            else:
                print(f"   前20个唯一值: {list(unique_values[:20])}")
                print(f"   ... 还有 {len(unique_values) - 20} 个唯一值")
    
    # 3. 分析空间坐标信息
    print("\n=== 空间坐标信息 ===")
    spatial_keys = ['spatial', 'X_spatial']
    for key in spatial_keys:
        if key in adata.obsm:
            coords = adata.obsm[key]
            print(f"\n📍 [{key}] 空间坐标:")
            print(f"   坐标形状: {coords.shape}")
            print(f"   X坐标范围: [{np.min(coords[:, 0]):.2f}, {np.max(coords[:, 0]):.2f}]")
            print(f"   Y坐标范围: [{np.min(coords[:, 1]):.2f}, {np.max(coords[:, 1]):.2f}]")
            print(f"   坐标示例 (前5行):\n{coords[:5]}")
    
    # 4. 组合字段分析（用于更细粒度的分割）
    print("\n=== 组合字段分析 ===")
    combination_fields = [
        ('slice', 'tissue'),
        ('slice_id', 'clust_annot'),
        ('fov', 'clust_annot')
    ]
    
    for field1, field2 in combination_fields:
        if field1 in adata.obs.columns and field2 in adata.obs.columns:
            combined = adata.obs[field1].astype(str) + "_" + adata.obs[field2].astype(str)
            unique_combined = combined.unique()
            print(f"\n🔗 [{field1} + {field2}] 组合字段:")
            print(f"   组合后唯一值数量: {len(unique_combined)}")
            if len(unique_combined) <= 20:
                print(f"   所有组合值: {list(unique_combined)}")
            else:
                print(f"   前10个组合值: {list(unique_combined[:10])}")
    
    # 5. 显示一些统计信息
    print("\n=== 数据统计 ===")
    print(f"总细胞数: {adata.n_obs}")
    print(f"总基因数: {adata.n_vars}")
    
    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
        print(f"空间密度 (细胞/单位面积): {adata.n_obs / ((np.max(coords[:, 0]) - np.min(coords[:, 0])) * (np.max(coords[:, 1]) - np.min(coords[:, 1]))):.4f}")

    print("\n✅ 组织区域语义分割数据分析完成!")

if __name__ == "__main__":
    # 请根据实际情况修改h5ad文件路径
    H5AD_FILE_PATH = "Allen2022Molecular_aging_MsBrainAgingSpatialDonor_2_1.h5ad"  # 或者是 kidney.h5ad 等其他文件名
    analyze_tissue_segmentation_data(H5AD_FILE_PATH)