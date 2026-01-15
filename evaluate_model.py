import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm
import torch
import sys
import os
import warnings

# 忽略 Scanpy 的部分警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 引用你的 model_engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_engine import NicheformerEngine

class NicheformerEvaluator:
    def __init__(self, h5ad_path, model_path, cell_type_col="cell_type", region_col="region"):
        self.engine = NicheformerEngine()
        self.engine.load_data(h5ad_path)
        self.engine.load_model(model_path)
        
        # 加载下游分类器
        self.engine.load_downstream_models()
        
        self.adata = self.engine.adata
        self.cell_type_col = cell_type_col
        self.region_col = region_col
        
        # 预计算 Embeddings
        if self.engine.embeddings_cache is None:
            self.engine._precompute_embeddings()

    def evaluate_cell_type_annotation(self):
        print("\n" + "="*40)
        print("📊 评估任务 1: 细胞类型注释 (Cell Type Annotation)")
        
        if self.cell_type_col not in self.adata.obs:
            print(f"❌ 错误: adata.obs 中找不到列 '{self.cell_type_col}'")
            print(f"ℹ️  可用列名: {list(self.adata.obs.columns)}")
            return

        try:
            pred_ids, legend = self.engine.predict_cell_types()
            if len(pred_ids) == 0: 
                print("⚠️  预测结果为空，跳过评估")
                return
                
            true_labels = self.adata.obs[self.cell_type_col].values.astype(str)
            
            # 映射 ID -> Name
            id_to_name = {item['id']: str(item['name']) for item in legend}
            pred_names = [id_to_name.get(pid, "Unknown") for pid in pred_ids]
            
            acc = accuracy_score(true_labels, pred_names)
            f1 = f1_score(true_labels, pred_names, average='weighted')
            
            print(f"✅ Accuracy : {acc:.4f}")
            print(f"✅ F1-Score : {f1:.4f}")
        except Exception as e:
            print(f"❌ 评估出错: {e}")

    def evaluate_tissue_segmentation(self):
        print("\n" + "="*40)
        print("📊 评估任务 2: 组织区域分割 (Tissue Segmentation)")
        
        if self.region_col not in self.adata.obs:
            print(f"❌ 错误: adata.obs 中找不到列 '{self.region_col}'")
            print(f"ℹ️  请检查下面的可用列名，并修改代码中的 GT_REGION 变量:")
            print(f"👉 {list(self.adata.obs.columns)}")
            return

        try:
            pred_ids, region_names_list = self.engine.segment_tissue_regions()
            true_regions = self.adata.obs[self.region_col].values.astype(str)
            
            # 将 region_names_list 里的元素也转为 str 防止类型不匹配
            region_names_list = [str(x) for x in region_names_list]
            pred_region_names = [region_names_list[rid] for rid in pred_ids]
            
            acc = accuracy_score(true_regions, pred_region_names)
            f1 = f1_score(true_regions, pred_region_names, average='weighted')
            
            print(f"✅ Accuracy : {acc:.4f}")
            print(f"✅ F1-Score : {f1:.4f}")
        except Exception as e:
            print(f"❌ 评估出错: {e}")

    def evaluate_zero_shot_clustering(self, n_clusters=10):
        print("\n" + "="*40)
        print(f"📊 评估任务 3: 零样本聚类 (Zero-shot Clustering, K={n_clusters})")
        
        if self.cell_type_col not in self.adata.obs:
            print(f"❌ 错误: 找不到参照列 '{self.cell_type_col}'")
            return
        
        try:
            cluster_labels, _ = self.engine.run_zero_shot_clustering(n_clusters=n_clusters)
            true_labels = self.adata.obs[self.cell_type_col].values
            
            ari = adjusted_rand_score(true_labels, cluster_labels)
            nmi = normalized_mutual_info_score(true_labels, cluster_labels)
            
            print(f"✅ ARI (Adjusted Rand Index)       : {ari:.4f}")
            print(f"✅ NMI (Normalized Mutual Info)   : {nmi:.4f}")
        except Exception as e:
            print(f"❌ 聚类评估出错: {e}")

    def evaluate_gene_imputation(self, n_test_genes=50):
        print("\n" + "="*40)
        print(f"📊 评估任务 4: 基因插补 (Gene Imputation, Top {n_test_genes} Genes)")
        
        # --- 【修复核心】更稳健的基因选择逻辑 ---
        hv_genes = []
        try:
            # 1. 尝试使用 Scanpy 高变基因 (可能会崩溃)
            print("尝试选取高变基因...")
            # 创建临时对象以免修改原数据
            temp_adata = self.adata.copy()
            # 如果是整数，说明可能是 Raw Counts，使用 seurat_v3 flavor
            if np.issubdtype(temp_adata.X.dtype, np.integer):
                sc.pp.highly_variable_genes(temp_adata, n_top_genes=n_test_genes, flavor='seurat_v3')
            else:
                # 否则先 Log 再算
                sc.pp.log1p(temp_adata)
                sc.pp.highly_variable_genes(temp_adata, n_top_genes=n_test_genes)
            
            hv_genes = temp_adata.var[temp_adata.var['highly_variable']].index.tolist()
            del temp_adata # 释放内存
            
        except Exception as e:
            print(f"⚠️ Scanpy 高变基因计算失败 ({str(e)})，切换到后备方案...")
            
        # 2. 后备方案：如果没有选出基因（或崩溃），则选平均表达量最高的基因
        if len(hv_genes) < n_test_genes:
            print("👉 使用平均表达量最高的基因作为测试集 (Fallback Strategy)")
            # 计算平均表达量
            if hasattr(self.adata.X, 'toarray'):
                means = np.array(self.adata.X.mean(axis=0)).flatten()
            else:
                means = np.array(self.adata.X.mean(axis=0)).flatten()
                
            # 获取 Top N 索引
            top_indices = np.argsort(means)[-n_test_genes:]
            hv_genes = self.adata.var_names[top_indices].tolist()
        
        hv_genes = hv_genes[:n_test_genes]
        print(f"已选择测试基因: {hv_genes[:5]} ...")
            
        print(f"已选择测试基因: {hv_genes[:5]} ...")
            
        pearson_list = []
        rmse_list = []
        mae_list = []
        
        # 🟢【修改开始】复制替换下面的循环块
        for gene in tqdm(hv_genes):
            try:
                # 1. 获取预测值 (Softplus 输出)
                pred_vals = self.engine.predict_gene_expression(gene)

                # 2. 获取真实值
                if isinstance(self.adata[:, gene].X, np.ndarray):
                    true_vals = self.adata[:, gene].X.flatten()
                else:
                    true_vals = self.adata[:, gene].X.toarray().flatten()

                # 3. 对数化 (Log1p) 以进行公平比较
                true_vals_log = np.log1p(true_vals)
                pred_vals_log = np.log1p(pred_vals) 
                
                # 4. 计算 Pearson 相关系数
                corr, _ = pearsonr(true_vals_log, pred_vals_log)
                
                # 🚨【关键修复】之前少了这一行，导致列表为空！
                if not np.isnan(corr):
                    pearson_list.append(corr)
                
                # 5. 计算 RMSE & MAE (建议统一用 Log 值比较)
                rmse = np.sqrt(mean_squared_error(true_vals_log, pred_vals_log))
                mae = mean_absolute_error(true_vals_log, pred_vals_log)

                rmse_list.append(rmse)
                mae_list.append(mae)

            except Exception as e_inner:
                # 🚨【调试增强】打印具体错误，不再当“哑巴”
                print(f"⚠️ 基因 {gene} 计算出错: {e_inner}")
                continue 
        # 🔴【修改结束】
        
        # 汇总结果
            
        # 汇总结果
        if len(pearson_list) > 0:
            avg_pearson = np.mean(pearson_list)
            avg_rmse = np.mean(rmse_list)
            avg_mae = np.mean(mae_list)
            
            print(f"✅ Pearson Correlation : {avg_pearson:.4f} (越高越好)")
            print(f"✅ RMSE (Normalized)   : {avg_rmse:.4f} (越低越好)")
            print(f"✅ MAE (Normalized)    : {avg_mae:.4f} (越低越好)")
        else:
            print("❌ 无法计算有效指标 (所有基因均返回 NaN)")

if __name__ == "__main__":
    # --- 配置区域 ---
    H5AD_FILE = "train.h5ad" 
    MODEL_PATH = "nicheformer_weights.pth"
    
    # ⚠️ 请在这里修改列名！⚠️
    # 如果不知道，先运行一次脚本，看 "评估任务 2" 的报错信息里会列出可用列名
    GT_CELL_TYPE = "cell_type"   # 你的细胞类型列名
    GT_REGION = "clust_annot"         # 你的区域列名 (可能是 'tissue', 'domain' 等)
    # ----------------
    
    evaluator = NicheformerEvaluator(H5AD_FILE, MODEL_PATH, GT_CELL_TYPE, GT_REGION)
    
    evaluator.evaluate_cell_type_annotation()
    evaluator.evaluate_tissue_segmentation()
    evaluator.evaluate_zero_shot_clustering(n_clusters=15)
    evaluator.evaluate_gene_imputation(n_test_genes=50)