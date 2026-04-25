import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

_calc = Calculator(descriptors, ignore_3D=True)

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    result = _calc(mol)
    row = {}
    for desc, val in zip(_calc.descriptors, result):
        try:
            row[str(desc)] = float(val)
        except:
            row[str(desc)] = float("nan")

    return pd.DataFrame([row])