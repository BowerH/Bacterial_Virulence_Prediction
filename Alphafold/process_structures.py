import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb

def extract_features(pdb_path: str) -> dict:
    """Extract structural features from AlphaFold PDB files"""
    ppdb = PandasPdb().read_pdb(pdb_path)
    atom_df = ppdb.df['ATOM']
    
    # Calculate secondary structure percentages
    helix = atom_df[atom_df['residue_name'] == 'HEL'].shape[0]
    sheet = atom_df[atom_df['residue_name'] == 'SHE'].shape[0]
    total = atom_df.shape[0]
    
    # Calculate solvent accessibility (dummy example)
    solvent_acc = atom_df['b_factor'].mean()  # Placeholder
    
    return {
        'alpha_helix': helix / total,
        'beta_sheet': sheet / total,
        'solvent_accessibility': solvent_acc,
        'plddt_score': ppdb.df['OTHERS'][ppdb.df['OTHERS']['record_name']=='ENDMDL']['b_factor'].mean()
    }

# Process all predictions
features = []
for pdb_file in os.listdir("data/processed/structures"):
    if pdb_file.endswith(".pdb"):
        feats = extract_features(os.path.join("data/processed/structures", pdb_file))
        features.append(feats)

# Save structural features
pd.DataFrame(features).to_csv("data/processed/structural_features.csv", index=False)