# 🦠 Protein Structure-Based Virulence Prediction

A working Repo for CSE7850 Project, Predicting bacterial virulence and pathogenicity from protein folding patterns using machine learning algorithms

## 📁 Repository Structure

```GeneratingVirulentData.ipynb```: Downloads and processes virulent protein sequences from VFDB, converts RefSeq IDs to UniProt IDs, and retrieves AlphaFold structures.

```GeneratingNonVirulentData.ipynb```: Collects non-virulent protein sequences from non-pathogenic organisms (e.g., Bacillus subtilis), processes them similarly, and prepares them for feature extraction.

```MLModelTraining.ipynb```: Loads extracted features, performs preprocessing, and trains machine learning models to classify proteins as virulent or non-virulent.

## 🧬 Workflow Overview

#### Data Collection:

- Virulent sequences sourced from VFDB

- Non-virulent sequences from UniProt (e.g., B. subtilis)

#### Structure Retrieval:

- UniProt IDs mapped to AlphaFoldDB entries

- 3D structure files (.pdb) downloaded automatically

#### Feature Extraction:

- Secondary structure content (α-helix, β-sheet, coil)

- Hydrogen bond density via DSSP

- Confidence metrics (mean pLDDT)

#### Model Training:

- Feature matrices combined and labeled (1 = virulent, 0 = non-virulent)

- Classifier trained with stratified sampling and evaluated on a held-out test set


## ⚙️ Dependencies

Install via ```conda```:

```
conda create -n virulence-prediction python=3.10
conda activate virulence-prediction
conda install -c conda-forge biopython pandas jupyter scikit-learn
pip install tqdm torch esm biopandas
```

Also required:

AlphaFoldDB access (public download)

DSSP executable (```mkdssp```, via ```conda install -c salilab dssp```)

## 📊 Output

```virulent_features.csv``` / ```nonvirulent_features.csv```: Structural features extracted from PDBs

Trained model and performance metrics (accuracy, F1-score, etc.)

Ready-to-use classifier for predicting new protein sequences


## 🚀 How to run

Clone this repo:

```
git clone https://github.com/your-username/virulence-structure-prediction.git
cd virulence-structure-prediction
```

Run the notebooks in order:

- ```Scripts/GeneratingVirulentData.ipynb```
- ```Scripts/GeneratingNonVirulentData.ipynb```
- ```Scripts/MLModelTraining.ipynb```

## Github Layout

```
Bacterial_Virulence_Prediction/
├── Data/
│   ├── structure_features_extracted_clean.csv                     # Non-virulent protein features
│   ├── structure_features_extracted_clean_final.csv               # Virulent protein features
├── Scripts/
│   ├── GeneratingVirulentData.ipynb           # Downloads and processes virulent protein data
│   ├── GeneratingNonVirulentData.ipynb        # Downloads and processes non-virulent protein data
│   ├── Model_Training.ipynb                   # Trains classification models
├── Model/
│   ├── ML_final_ensemble_model.pkl            # Trained ensemble model (e.g. RandomForest + XGBoost)
├── Attachments/
│   └── ML_final_model_comparison.csv          # Model benchmarking results
├── README.md                                  # Project overview and usage instructions
└── .gitignore                                 # Ignore logs, checkpoints, cache files, etc.
```

## 👥 Contributors

| Name | GitHub | 
|------|--------|
| <img src="https://github.com/sanyabadole.png" width="50"/> | [@sanyabadole](https://github.com/sanyabadole) 
| <img src="https://github.com/BowerH.png" width="50"/> | [@BowerH](https://github.com/BowerH) 
| <img src="https://github.com/binfwizard.png" width="50"/> | [@binfwizard](https://github.com/binfwizard) 
| <img src="[https://github.gatech.edu/rfatma3).png" width="50"/> | [@rfatma3](https://github.gatech.edu/rfatma3) 


## 📌 Citation

If you use this pipeline in your work, please cite the repository and relevant data sources (VFDB, AlphaFoldDB, UniProt).
