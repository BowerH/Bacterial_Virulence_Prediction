
# pip install pandas biopython transformers torch pyarrow
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
import torch


# ## 1. Configuration
config = {
    "embedding_model": "Rostlab/prot_bert",  # ProtBERT model for embeddings
    "max_sequence_length": 1024,            # Maximum sequence length for embeddings
    "input_files": {
        "vfdb": "data/raw/vfdb_sequences.fasta",
        "uniprot": "data/raw/uniprot_ecoli.csv"
    },
    "output_files": {
        "processed_data": "data/processed/processed_sequences.csv",
        "embeddings": "data/processed/sequence_embeddings.parquet"
    }
}

# ## 2. Data Cleaning
def clean_sequence(sequence: str) -> str:
    """
    Clean protein sequence by removing unnatural amino acids.
    """
    # Replace rare amino acids with 'X'
    sequence = re.sub(r"[UZOB]", "X", sequence)
    return sequence

def load_vfdb(file_path: str) -> pd.DataFrame:
    """
    Load VFDB dataset from FASTA file.
    """
    from Bio import SeqIO
    records = []
    for record in SeqIO.parse(file_path, "fasta"):
        records.append({
            "id": record.id,
            "sequence": clean_sequence(str(record.seq)),
            "source": "VFDB",
            "label": 1  # Virulent
        })
    return pd.DataFrame(records)

def load_uniprot(file_path: str) -> pd.DataFrame:
    """
    Load UniProt dataset from CSV file.
    """
    df = pd.read_csv(file_path)
    df['sequence'] = df['Sequence'].apply(clean_sequence)
    return df[['Entry', 'sequence']].rename(columns={"Entry": "id"}).assign(label=0)

# Load datasets
vfdb_df = load_vfdb(config["input_files"]["vfdb"])
uniprot_df = load_uniprot(config["input_files"]["uniprot"])

# Combine datasets and shuffle
full_df = pd.concat([vfdb_df, uniprot_df]).sample(frac=1).reset_index(drop=True)

# Save cleaned data
full_df.to_csv(config["output_files"]["processed_data"], index=False)
print("Data cleaning completed and saved.")

# ## 3. Sequence Embedding with ProtBERT
class ProtBERTEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"])
        self.model = AutoModel.from_pretrained(config["embedding_model"])
    
    def embed(self, sequence: str) -> torch.Tensor:
        """
        Generate ProtBERT embeddings for a single protein sequence.
        """
        inputs = self.tokenizer(sequence, return_tensors="pt", 
                                max_length=config["max_sequence_length"],
                                truncation=True,
                                padding="max_length")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the [CLS] token embedding as the sequence representation
        return outputs.last_hidden_state[:, 0, :].squeeze()

def generate_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate embeddings for all sequences in the dataset.
    """
    embedder = ProtBERTEmbedder()
    
    embeddings = []
    for seq in df['sequence']:
        embedding = embedder.embed(seq).numpy()
        embeddings.append(embedding)
    
    # Convert embeddings to DataFrame
    embedding_df = pd.DataFrame(embeddings)
    
    # Add labels for supervised learning
    embedding_df['label'] = df['label']
    
    return embedding_df

# Generate embeddings
embedding_df = generate_embeddings(full_df)

# Save embeddings to file
embedding_df.to_parquet(config["output_files"]["embeddings"])
print("Sequence embeddings generated and saved.")


# ## 4. Structural Feature Extraction (Optional)
def extract_structural_features(pdb_file: str) -> dict:
    """
    Extract structural features such as alpha helices and beta sheets from PDB files.
    This step requires AlphaFold predictions to be available.
    
    Placeholder implementation; replace with actual feature extraction logic.
    """
    features = {
        'alpha_helix_percent': 0.0,
        'beta_sheet_percent': 0.0,
        'solvent_accessibility': 0.0,
        'distance_map': None  # Example placeholder for distance map matrix
    }
    
    return features

print("Preprocessing completed successfully.")