# -*- coding: utf-8 -*-
"""
Created on Wed May 21 20:35:45 2025

@author: yqz
"""

from SmilesEnumerator import SmilesEnumerator

sme = SmilesEnumerator()
smiles_synonymous_list=[]
for i in range(200):

    smiles_will_add=sme.randomize_smiles("CCC(=O)O[C@@]1(CC[NH+](C[C@H]1CC=C)C)c2ccccc2")
    smiles_synonymous_list.append(smiles_will_add)

smiles_synonymous_list = list(set(smiles_synonymous_list))

print(len(smiles_synonymous_list))
