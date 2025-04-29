# Bacterial_Virulence_Prediction

A working Repo for CSE7850 Project, Predicting bacterial virulence and pathogenicity from protein folding patterns using machine learning algorithms


## How to run

#### 1. Make sure you have all dependencies installed:

```
pip install pandas numpy matplotlib seaborn scikit-learn biopython torch fair-esm xgboost lightgbm catboost scikit-optimize joblib colabfold
```

#### 2. Fetch Data

For ease of use, we provided a dataset that we used for our report. These are two files located in the Data/ folder.

If you are looking to run this on a different dataset, you must follow these steps:
    1.
    2.
    3.

#### 3. Train the model

```
python train_model.py \
  --embeddings data/processed/sequence_embeddings.parquet \
  --structure data/processed/structural_features.csv \
  --output models/virulence_prediction
```

#### 4. Run Final Analysis and visualizations

```
jupyter notebook Notebooks/main_analysis.ipynb
```

## Github Layout

```
Bacterial_Virulence_Prediction/
├── Data/
│   ├── structure_features_extracted_clean.csv                     # Processed CSV file for non virulent protiens
│   ├── structure_features_extracted_clean_final.csv               # Processed CSV file for virulent protiens
├── Scripts/
│   ├── Generate_Virulent_CSV.ipynb           # Script to generate virulent CSV file
│   └── Model_Training.ipynb                  # Model training script
├── environment.yml                           # Conda environment file
├── README.md                    
└── .gitignore                                # Files to exclude from Git tracking
```

