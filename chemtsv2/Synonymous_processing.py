import re
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize



def tokenize_smiles(smiles_list, use_selfies=False):
    tokenized_smiles_list = []
    unique_token_set = set()
    for smi in smiles_list:
        tokenized_smiles =  smi_tokenizer(smi)
        tokenized_smiles.append("\n")
        unique_token_set |= set(tokenized_smiles)
        tokenized_smiles_list.append(tokenized_smiles)
    return sorted(list(unique_token_set)), tokenized_smiles_list



def smi_tokenizer(smi):
    """
    This function is based on https://github.com/pschwllr/MolecularTransformer#pre-processing
    Modified by Shoichi Ishida
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens)
    tokens.insert(0, "&")
    return tokens



def standardize_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        mol = rdMolStandardize.FragmentRemover().remove(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return smi


