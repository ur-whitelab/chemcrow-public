# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:06:19 2023

@author: Alex
"""
import os, sys
from chemcrow import *


chem_model = ChemCrow(model="./models/llama-2-7b.Q8_0.gguf", 
                      tools_model="./models/llama-2-7b.Q8_0.gguf", 
                      temp=0.1, verbose=False, max_tokens=100, n_ctx=2048)
output = chem_model.run("What is the molecular weight of tylenol?")

print(output)


# =============================================================================
# chem_model = ChemCrow(model_path="./models/llama-2-7b-chat.ggmlv3.q4_0.bin", verbose=False)
# x = chem_model.run("What is the molecular weight of tylenol?")
# 
# print(x)
# 
# =============================================================================
