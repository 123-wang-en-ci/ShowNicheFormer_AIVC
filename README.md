

# Nicheformer 3D: Spatial Transcriptomics Visualization System

This system is an interactive spatial transcriptomics visualization system based on Unity and Deep Learning. It combines a high-performance 3D rendering engine with an advanced AI model (Nicheformer) to provide researchers with an intuitive platform to explore cell distribution, gene expression, cell type annotation, and tissue region segmentation.

## The main function

The system integrates five core analysis modules:

- **Interactive 3D View**: Render tens of thousands of cells in three-dimensional space in real time to visually display tissue structure.  -
- **Gene Expression and Imputation (Gene Imputation)**: Query the expression level of specific genes in real time, and use the AI model to interpolate sparse data to restore the true expression pattern. 
- **Auto Annotation**: Automatically predict cell types based on the Nicheformer model and generate a visual legend.  
-  **Tissue Segmentation**: Automatically identify and segment different tissue functional areas (such as cortex, medulla, etc.), supporting independent display of single areas.  
- -**Zero-Shot Clustering**: User-defined number of clusters, potential cell subpopulations can be discovered instantly without retraining.

## Quick start

### 1. Environmental preparation

Backend (Python):

Make sure you have the Python environment installed and the necessary dependencies.

```
pip install -r requirements.txt
```

The weights and data needed in the code can be found in[ShowNicheFormer](https://huggingface.co/datasets/www123222/ShowNicheFormer/tree/main)

**Front-end (Unity):**

- Install Unity Hub and Unity Editor (2020.3 or higher recommended).  -
- Assign the current project's Scripts folder

### 2. Running steps

1. **Launch Backend Services**:

   ```
   python server.py
   # Runs by default http://127.0.0.1:8000
   ```

2. **Start Unity client**:

   - Open the project in the Unity Editor.  
   - Open the MainScene.  
   -  Click the **Play** button at the top.

