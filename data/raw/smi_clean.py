from pathlib import Path
import pandas as pd
import argparse 

def main(smiles_file):
    try:
        smiles_file_path = Path(smiles_file)
        if not smiles_file_path.is_file():
            raise FileNotFoundError("File not found. Please provide a valid SMILES file.")
    
        smiles = pd.read_csv(smiles_file_path, header=None, delim_whitespace=True)
        smiles[0].to_csv(Path.cwd().parent / 'clean' / f'{smiles_file_path.stem}_clean.smi',header=None, index=None)
        
        print("Program finished!")
    except Exception as e:
        print("Error:", e)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove mol names from SMILES file.')
    parser.add_argument('smiles_file', type=str, help='The the SMILES file')
    args = parser.parse_args()
    
    # Call main function with provided SMILES file name
    main(args.smiles_file)
    
    
    