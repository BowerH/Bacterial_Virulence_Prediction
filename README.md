# Bacterial_Virulence_Prediction

A working Repo for CSE7850 Project, Predicting bacterial virulence and pathogenicity from protein folding patterns using machine learning algorithms


## How to run

#### 1. Make sure you have all dependencies installed:

```
pip install pandas biopython transformers torch pyarrow numpy matplotlib tensorflow scikit-learn transformers
```

#### 2. Pre process data

```
python preprocessing.py
```

#### 3. Submit AlphaFold job to HPC

Note: This MUST be run on an NVDIA A100 GPU

```
sbatch alphafold/run_alphafold.slurm
```

#### 4. Process Alphafold Structures

```
python process_structures.py
```

#### 5. Train the model

```
python train_model.py \
  --embeddings data/processed/sequence_embeddings.parquet \
  --structure data/processed/structural_features.csv \
  --output models/virulence_prediction
```

#### 6. Run Final Analysis and visualizations

```
jupyter notebook Notebooks/main_analysis.ipynb
```

## Github Layout

```
project_root/
├── Notebooks/
│   ├── main_analysis.ipynb      # Primary Jupyter notebook
│   ├── temp.ipynb               # Optional exploratory notebooks
├── Data/
│   ├── Raw_Data/                # Raw datasets (VFDB, MvirDB,UniProt)
│   ├── processed/               # Processed datasets (embeddings, features)
├── Scripts/
│   ├── preprocessing.py            # Data preprocessing script
│   ├── train_model.py           # Model training script
├── Results/
│   ├── Figures/                 # ROC curves, heatmaps, etc.
│   ├── predictions.csv          # Model predictions
├── Alphafold/
│   ├── run_alphafold.slurm       # HPC submission script
│   └── process_structures.py 
├── environment.yml              # Conda environment file
├── README.md                    
└── .gitignore                   # Files to exclude from Git tracking
```

## CPU Requirements

NVDIA A100 GPU

