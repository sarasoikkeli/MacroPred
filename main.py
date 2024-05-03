from config import *
import logging
from pathlib import Path
import sys
import argparse
import pandas as pd
from rdkit import Chem
from mordred import Calculator
from sklearn.preprocessing import StandardScaler
from joblib import load
import numpy as np
import xgboost

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def calculate_descriptors(smiles_list):
    """Calculate molecular descriptors for a list of SMILES strings."""
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    calc = Calculator(BORUTA_FEATURES, ignore_3D=True)
    
    if None in mols:
        logger.error("Error in processing SMILES. Check the content of the file and re-run the program. Make sure to exclude column header.")
        sys.exit(1)
    
    logger.info("Calculating descriptors...")
    try:
        descs = calc.pandas(mols)
        # Check for missing or non-numeric values
        if descs.isnull().values.any() or not np.isfinite(descs.values).all():
            raise ValueError("Descriptors calculation resulted in missing or non-numeric values.")

        descs_file = Path.cwd() / 'descriptors.csv'
        descs.to_csv(descs_file, index=False)
        return descs
    except Exception as e:
        logger.error("Error in calculating descriptors: %s", str(e))
        sys.exit(1)
    
def predict_labels(descriptors):
    """Predict labels with the serialized model using descriptors as input."""
    model = load(MODEL)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(descriptors)
    X_scaled = pd.DataFrame(data=X_scaled, columns=scaler.get_feature_names_out())
    
    logger.info("Fitting the model...")
    return model.predict(X_scaled)
  

def main(smiles_file):
    """Main function to load SMILES file, calculate descriptors, and predict labels."""
    # Load SMILES file
    
    try:
        smiles_file_path = Path(smiles_file)
        if not smiles_file_path.is_file():
            raise FileNotFoundError("File not found. Please provide a valid SMILES file.")
        
        with open(smiles_file_path, 'r') as file:
            smiles_list = file.read().strip().splitlines()
        
        if not smiles_list:
            raise ValueError("SMILES file is empty.")
        
        # Calculate descriptors
        descriptors = calculate_descriptors(smiles_list)

        # Predict labels
        labels = predict_labels(descriptors)

        # Create DataFrame with SMILES and predicted labels
        results_df = pd.DataFrame({'SMILES': smiles_list, 'Predicted_Label': labels})
        results_file = Path.cwd() / f'predicted_labels_{smiles_file_path.stem}.csv'
        results_df.to_csv(results_file, index=False)
        
        
        logger.info("Program finished.")
        logger.info("Results saved to: %s", results_file)
        
    except FileNotFoundError as e:
        logger.error(str(e))
    except ValueError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error("Error: %s", str(e))

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculate descriptors and predict labels from a SMILES file.')
    parser.add_argument('smiles_file', type=str, help='The the SMILES file')
    args = parser.parse_args()

    # Call main function with provided SMILES file name
    main(args.smiles_file)
