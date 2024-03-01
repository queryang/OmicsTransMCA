#!/usr/bin/env python3
"""Train PaccMann predictor."""
import json
import os
import pickle
from copy import deepcopy
from time import time
import numpy as np
import pandas as pd
import torch
from pytoda.smiles.smiles_language import SMILESTokenizer
from OmicsTransMCA_predictor.models import MODEL_FACTORY
from OmicsTransMCA_predictor.utils.hyperparams import OPTIMIZER_FACTORY
from OmicsTransMCA_predictor.utils.loss_functions import pearsonr, r2_score
from OmicsTransMCA_predictor.utils.utils import get_device, get_log_molar
from model_omics_experiment.tools.OmicsDrugSensitivityDataset_GEP import OmicsDrugSensitivityDataset_GEP
from model_omics_experiment.tools.OmicsDrugSensitivityDataset_GEP_CNV_MUT import OmicsDrugSensitivityDataset_GEP_CNV_MUT

def main(
    test_sensitivity_filepath,
    gep_filepath,
    smi_filepath,
    gene_filepath,
    smiles_language_filepath,
    model_path,
    params_filepath,
):

    # Process parameter file:
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))
        params.update(
            {
                "num_workers": 0
            }
        )
    print(params)

    # Prepare the dataset
    print("Start data preprocessing...")

    # Load SMILES language
    smiles_language = SMILESTokenizer.from_pretrained(smiles_language_filepath)
    smiles_language.set_encoding_transforms(
        add_start_and_stop=params.get("add_start_and_stop", True),
        padding=params.get("padding", True),
        padding_length=smiles_language.max_token_sequence_length,
    )
    test_smiles_language = deepcopy(smiles_language)
    smiles_language.set_smiles_transforms(
        augment=params.get("augment_smiles", False),
        canonical=params.get("smiles_canonical", False),
        kekulize=params.get("smiles_kekulize", False),
        all_bonds_explicit=params.get("smiles_bonds_explicit", False),
        all_hs_explicit=params.get("smiles_all_hs_explicit", False),
        remove_bonddir=params.get("smiles_remove_bonddir", False),
        remove_chirality=params.get("smiles_remove_chirality", False),
        selfies=params.get("selfies", False),
        sanitize=params.get("selfies", False),
    )
    test_smiles_language.set_smiles_transforms(
        augment=False,
        canonical=params.get("test_smiles_canonical", False),
        kekulize=params.get("smiles_kekulize", False),
        all_bonds_explicit=params.get("smiles_bonds_explicit", False),
        all_hs_explicit=params.get("smiles_all_hs_explicit", False),
        remove_bonddir=params.get("smiles_remove_bonddir", False),
        remove_chirality=params.get("smiles_remove_chirality", False),
        selfies=params.get("selfies", False),
        sanitize=params.get("selfies", False),
    )

    # Load the gene list
    with open(gene_filepath, "rb") as f:
        pathway_list = pickle.load(f)

    # Load the datasets
    test_dataset = OmicsDrugSensitivityDataset_GEP(
        drug_sensitivity_filepath=test_sensitivity_filepath,
        smiles_filepath=smi_filepath,
        gep_filepath=gep_filepath,
        gep_standardize=params.get("gep_standardize", False),
        smiles_language=smiles_language,
        drug_sensitivity_min_max=params.get("drug_sensitivity_min_max", True),
        iterate_dataset=False,
    )
    min_value = test_dataset.drug_sensitivity_processing_parameters['parameters']['min']
    max_value = test_dataset.drug_sensitivity_processing_parameters['parameters']['max']
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=params.get("num_workers", 4),
    )
    print(
        f'Test dataset has {len(test_dataset)} samples with {len(test_loader)} batches'
    )

    device = get_device()

    print(
        f'model is {device}'
    )

    model_name = params.get("model_fn", "mca_GEP")
    model = MODEL_FACTORY[model_name](params).to(device)
    model._associate_language(smiles_language)

    try:
        print(f'Attempting to restore model from {model_path}...')
        model.load(model_path, map_location=device)
    except Exception:
        raise ValueError(f'Error in restoring model from {model_path}!')

    model.eval()
    with torch.no_grad():
        test_loss = 0
        log_pres = []
        log_labels = []
        gene_attentions = []
        # cnv_attentions = []
        # mut_attentions = []
        smiles_attentions_geps = []
        # smiles_attentions_cnvs = []
        # smiles_attentions_muts = []
        for ind, (smiles, gep, y) in enumerate(test_loader):
            y_hat, pred_dict = model(
                torch.squeeze(smiles.to(device)), gep.to(device)
            )
            log_pre = pred_dict.get("log_micromolar_IC50")
            log_pres.append(log_pre)
            gene_attention = pred_dict.get("gene_attention")
            # 通过取均值的方法将gene_attention维度从[192,619,5]->[192,619]
            gene_attention = torch.mean(gene_attention, dim=2)
            gene_attentions.append(gene_attention)
            # cnv_attention = pred_dict.get("cnv_attention")
            # cnv_attention = torch.mean(cnv_attention, dim=2)
            # cnv_attentions.append(cnv_attention)
            # mut_attention = pred_dict.get("mut_attention")
            # mut_attention = torch.mean(mut_attention, dim=2)
            # mut_attentions.append(mut_attention)
            smiles_attention_gep = pred_dict.get("smiles_attention_gep")
            smiles_attention_gep = torch.mean(smiles_attention_gep, dim=2)
            smiles_attentions_geps.append(smiles_attention_gep)
            # smiles_attention_cnv = pred_dict.get("smiles_attention_cnv")
            # smiles_attention_cnv = torch.mean(smiles_attention_cnv, dim=2)
            # smiles_attentions_cnvs.append(smiles_attention_cnv)
            # smiles_attention_mut = pred_dict.get("smiles_attention_mut")
            # smiles_attention_mut = torch.mean(smiles_attention_mut, dim=2)
            # smiles_attentions_muts.append(smiles_attention_mut)

            log_y = get_log_molar(y, ic50_max=max_value, ic50_min=min_value)
            log_labels.append(log_y)
            # labels.append(y)
            loss = model.loss(log_pre, log_y.to(device))
            test_loss += loss.item()

    # on the logIC50 scale
    predictions = torch.cat([p.cpu() for preds in log_pres for p in preds])
    labels = torch.cat([l.cpu() for label in log_labels for l in label])
    # 计算pearson相关系数
    test_pearson = pearsonr(torch.Tensor(predictions), torch.Tensor(labels))
    test_rmse = torch.sqrt(torch.mean((predictions - labels) ** 2))
    test_loss = test_loss / len(test_loader)
    test_r2 = r2_score(torch.Tensor(predictions), torch.Tensor(labels))
    print(
        f"\t **** TESTING **** \n"
        f"MSE: {test_loss:.5f}, "
        f"Pearson: {test_pearson:.5f}, "
        f"RMSE: {test_rmse:.5f}, "
        f"R2: {test_r2:.5f}. "
    )
    #将gene_attentions和smiles_attentions_geps保存为csv文件
    gene_attentions = torch.cat([gene_attentions[i] for i in range(len(gene_attentions))])
    smiles_attentions_geps = torch.cat([smiles_attentions_geps[i] for i in range(len(smiles_attentions_geps))])

    gene_attentions = gene_attentions.cpu().numpy()
    smiles_attentions_geps = smiles_attentions_geps.cpu().numpy()

    gene_attentions = pd.DataFrame(gene_attentions)
    smiles_attentions_geps = pd.DataFrame(smiles_attentions_geps)

    gene_attentions.to_csv('attention_result/GEP_CellBlind_LUNG_gene_attention2.csv',index=False)
    smiles_attentions_geps.to_csv('attention_result/GEP_CellBlind_LUNG_smiles_attentions_gep2.csv',index=False)

if __name__ == "__main__":

    test_sensitivity_filepath = '../model_omics_experiment/data/drug_sensitivity_lung_CellBlind_test.csv'
    gep_filepath = '../model_omics_experiment/data/GeneExp_Wilcoxon_test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
    cnv_filepath = '../model_omics_experiment/data/CNV_Cardinality_analysis_of_variance_Latest_MEDICUS.csv'
    mut_filepath = '../model_omics_experiment/data/MUT_cardinality_analysis_of_variance_Only_MEDICUS.csv'
    smi_filepath = '../model_omics_experiment/data/ccle-gdsc.smi'
    gene_filepath = '../model_omics_experiment/data/MUDICUS_Omic_619_pathways.pkl'
    smiles_language_filepath = '../model_omics_experiment/data/smiles_language/tokenizer_customized'
    model_path = 'models/MCA_GEP_MEDICUS_CellBlind_LUNG/weights/best_mse_mca_GEP.pt'
    params_filepath = 'models/MCA_GEP_MEDICUS_CellBlind_LUNG/model_params.json'
    # training_name = 'TRANS_MCA_GEP(Log10_P_value)_CNV(Cardinality_Analysis)_MUT_MEDICUS619'
    # run the training
    main(
        test_sensitivity_filepath,
        gep_filepath,
        smi_filepath,
        gene_filepath,
        smiles_language_filepath,
        model_path,
        params_filepath,
        # training_name
    )
