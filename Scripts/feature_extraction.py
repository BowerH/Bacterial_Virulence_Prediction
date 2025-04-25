# feature_extraction.py
# Step 3: Feature Extraction for Predicting Staphylococcus Virulence

import os
import pandas as pd
import numpy as np
from biopandas.pdb import PandasPdb
import torch
import esm
from Bio import SeqIO
from Bio.PDB import PDBParser, DSSP


def extract_fasta_sequences(fasta_path):
    id_to_seq = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        header = record.id
        protein_id = header.split("(")[0]
        id_to_seq[protein_id] = str(record.seq)
    return id_to_seq

def find_pdbs(base_dir):
    """Map protein_id to PDB path based on prefix match"""
    pdb_dict = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".pdb") and "rank_001" in f:  # You can adjust rank if needed
                file_prefix = f.split("_")[0]  # e.g. VFG001274
                full_path = os.path.join(root, f)
                pdb_dict[file_prefix] = full_path
    return pdb_dict

def merge_fasta_pdb_to_csv(fasta_path, pdb_dir, output_csv):
    seqs = extract_fasta_sequences(fasta_path)
    pdbs = find_pdbs(pdb_dir)

    merged = []
    for pid, seq in seqs.items():
        if pid in pdbs:
            merged.append({
                "protein_id": pid,
                "sequence": seq,
                "pdb_path": pdbs[pid]
            })
        else:
            print(f"[!] Warning: No PDB found for {pid}")

    df = pd.DataFrame(merged)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} records to {output_csv}")

# === Paths ===
fasta_file = r"\\qrfileserver\QR-ANALYTICS\Hannah Bower\My stuff\vfdb_staph_cleaned.fasta"
alphafold_dir = r"\\qrfileserver\QR-ANALYTICS\Hannah Bower\My stuff\vfdb_alphafold\vfdb_alphafold"  # update this path to your actual folder
output_csv = "protein_input_dataset.csv"

merge_fasta_pdb_to_csv(fasta_file, alphafold_dir, output_csv)


def extract_structural_features(pdb_path: str) -> dict:
    #Extract structural features: β-sheet content, H-bond density, pLDDT score
    try:
        # Parse structure
        parser = PDBParser()
        structure = parser.get_structure("protein", pdb_path)
        model = structure[0]
        dssp = DSSP(model, pdb_path)

        # Extract β-sheet content
        ss_codes = [res[2] for res in dssp]  # DSSP codes: 'E' = β-sheet
        n_residues = len(ss_codes)
        beta_sheet_content = ss_codes.count('E') / n_residues if n_residues else 0

        # Extract pLDDT from B-factor field
        ppdb = PandasPdb().read_pdb(pdb_path)
        plddt = ppdb.df['ATOM']['b_factor'].mean()

        return {
            "beta_sheet_content": beta_sheet_content,
            "plddt_score": plddt,
            "hbond_density": None  # optional: skip or approximate if needed
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"beta_sheet_content": None, "plddt_score": None, "hbond_density": None}

def extract_esm_embedding(sequence: str, model_name="esm2_t33_650M_UR50D") -> dict:
    # Extract ESM-2 language model embeddings from protein sequence
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    embedding = token_representations[0, 1:len(sequence)+1].mean(0).numpy()

    return {f"esm2_feat_{i}": val for i, val in enumerate(embedding)}

def extract_all_features(sequence: str, pdb_path: str) -> dict:
    """Combine structural and language model features"""
    struct_feats = extract_structural_features(pdb_path)
    esm_feats = extract_esm_embedding(sequence)
    return {**struct_feats, **esm_feats}

def run_pipeline(csv_path: str, output_path: str):
    """Run feature extraction pipeline from dataset CSV"""
    df = pd.read_csv(csv_path)
    all_features = []

    for _, row in df.iterrows():
        protein_id, sequence, pdb_path = row['protein_id'], row['sequence'], row['pdb_path']
        feats = extract_all_features(sequence, pdb_path)
        feats['protein_id'] = protein_id
        all_features.append(feats)

    feat_df = pd.DataFrame(all_features)
    feat_df.to_csv(output_path, index=False)

# Usage example:
run_pipeline("protein_input_dataset.csv", "protein_virulence_features.csv")
