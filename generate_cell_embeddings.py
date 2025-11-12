import os
import pandas as pd
import numpy as np
import torch
from evo2 import Evo2
from transformers import AutoModel
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

rna_meta = pd.read_csv("/home/nebius/cellian/data/perturb-cite-seq/SCP1064/metadata/RNA_metadata.csv")
rna_exp = pd.read_csv("/home/nebius/cellian/data/perturb-cite-seq/SCP1064/other/RNA_expression.csv", index_col=0).T
prot_exp = pd.read_csv("/home/nebius/cellian/data/perturb-cite-seq/SCP1064/expression/Protein_expression.csv", index_col=0).T
rna_meta = rna_meta.rename(columns={"NAME": "cell"}).set_index("cell")

common = rna_meta.index.intersection(rna_exp.index).intersection(prot_exp.index)
rna_meta = rna_meta.loc[common]
rna_exp = rna_exp.loc[common]
prot_exp = prot_exp.loc[common]

gene_to_seq = {}  # supply dict mapping gene -> DNA seq

evo = Evo2('evo2_7b').to(device)
def evo_embed(seq):
    t = torch.tensor(evo.tokenizer.tokenize(seq), dtype=torch.int64).unsqueeze(0).to(device)
    _, e = evo(t, return_embeddings=True, layer_names=['blocks.28.mlp.l3'])
    return e['blocks.28.mlp.l3'].mean(dim=1).detach().cpu().numpy().squeeze()

perturb_embs = []
for g in rna_meta["sgRNA"].fillna("CTRL"):
    if g == "CTRL":
        perturb_embs.append(np.zeros(1024))
    else:
        gene = g.split("_")[0]
        if gene in gene_to_seq:
            perturb_embs.append(evo_embed(gene_to_seq[gene]))
        else:
            perturb_embs.append(np.zeros(1024))
perturb_embs = np.vstack(perturb_embs)

state = AutoModel.from_pretrained("arcinstitute/SE-600M").to(device)
rna_exp = rna_exp.apply(lambda x: np.log1p(x / (x.sum() + 1) * 1e4), axis=1)
rna_tensor = torch.tensor(rna_exp.values, dtype=torch.float32).to(device)
with torch.no_grad():
    se_out = state(inputs_embeds=rna_tensor.unsqueeze(1))
    rna_emb = se_out.last_hidden_state.mean(dim=1).detach().cpu().numpy()

captain_path = "/home/nebius/cellian/foundation_models/CAPTAIN_BASE/CAPTAIN_Base.pt"
captain_model = torch.load(captain_path, map_location=device)
rna_scaled = StandardScaler().fit_transform(rna_exp)
prot_scaled = StandardScaler().fit_transform(prot_exp)
rna_tensor = torch.tensor(rna_scaled, dtype=torch.float32).to(device)
prot_tensor = torch.tensor(prot_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    prot_emb = captain_model.encode(rna_tensor, prot_tensor).detach().cpu().numpy()

os.makedirs("/home/nebius/cellian/outputs", exist_ok=True)
torch.save({
    "perturb_emb": torch.tensor(perturb_embs, dtype=torch.float32),
    "rna_emb": torch.tensor(rna_emb, dtype=torch.float32),
    "prot_emb": torch.tensor(prot_emb, dtype=torch.float32),
    "cells": list(common)
}, "/home/nebius/cellian/outputs/cell_embeddings.pt")

