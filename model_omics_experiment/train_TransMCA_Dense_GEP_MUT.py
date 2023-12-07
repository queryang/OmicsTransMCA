#!/usr/bin/env python3
"""Train PaccMann predictor."""
import json
import os
import pickle
from copy import deepcopy
from time import time
import numpy as np
import torch
from OmicsTransMCA_predictor.models import MODEL_FACTORY
from OmicsTransMCA_predictor.utils.hyperparams import OPTIMIZER_FACTORY
from OmicsTransMCA_predictor.utils.loss_functions import pearsonr, r2_score
from OmicsTransMCA_predictor.utils.utils import get_device, get_log_molar
from pytoda.smiles.smiles_language import SMILESTokenizer

from model_omics_experiment.tools.OmicsDrugSensitivityDataset_GEP_MUT import OmicsDrugSensitivityDataset_GEP_MUT


def main(
    train_sensitivity_filepath,
    test_sensitivity_filepath,
    gep_filepath,
    mut_filepath,
    smi_filepath,
    gene_filepath,
    smiles_language_filepath,
    model_path,
    params_filepath,
    training_name
):

    # Process parameter file:
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))
    print(params)
    # Create model directory and dump files
    model_dir = os.path.join(model_path, training_name)
    os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "results"), exist_ok=True)
    with open(os.path.join(model_dir, "model_params.json"), "w") as fp:
        json.dump(params, fp, indent=4)

    # Prepare the dataset
    print("Start data preprocessing...")

    # Load SMILES language
    smiles_language = SMILESTokenizer.from_pretrained(smiles_language_filepath)
    smiles_language.set_encoding_transforms(
        add_start_and_stop=params.get("add_start_and_stop", True),
        padding=params.get("padding", True),
        padding_length=smiles_language.max_token_sequence_length,
        # padding_length=params.get("smiles_padding_length", None),
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
        augment=params.get("augment_smiles", False),
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
    # OmicsDrugSensitivityDataset 重写的数据集class
    train_dataset = OmicsDrugSensitivityDataset_GEP_MUT(
        drug_sensitivity_filepath=train_sensitivity_filepath,
        smiles_filepath=smi_filepath,
        gep_filepath=gep_filepath,
        mut_filepath=mut_filepath,
        gep_standardize=params.get("gep_standardize", False),
        mut_standardize=params.get("mut_standardize", False),
        smiles_language=smiles_language,
        drug_sensitivity_min_max=params.get("drug_sensitivity_min_max", True),
        iterate_dataset=False,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=params.get("num_workers", 4),
    )
    test_dataset = OmicsDrugSensitivityDataset_GEP_MUT(
        drug_sensitivity_filepath=test_sensitivity_filepath,
        smiles_filepath=smi_filepath,
        gep_filepath=gep_filepath,
        mut_filepath=mut_filepath,
        gep_standardize=params.get("gep_standardize", False),
        mut_standardize=params.get("mut_standardize", False),
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
        f"Training dataset has {len(train_dataset)} samples, test set has "
        f"{len(test_dataset)}."
    )

    device = get_device()

    save_top_model = os.path.join(model_dir, "weights/{}_{}_{}.pt")
    params.update(
        {  # yapf: disable
            "number_of_genes": len(pathway_list),
            "smiles_vocabulary_size": smiles_language.number_of_tokens,
            "drug_sensitivity_processing_parameters": train_dataset.drug_sensitivity_processing_parameters,
            "gene_expression_processing_parameters": {},
        }
    )
    model_name = params.get("model_fn", "trans_mca_dense_GEP_MUT")
    model = MODEL_FACTORY[model_name](params).to(device)
    model._associate_language(smiles_language)


    if os.path.isfile(os.path.join(model_dir, "weights", f"best_mse_{model_name}.pt")):
        print("Found existing model, restoring now...")
        model.load(os.path.join(model_dir, "weights", f"best_mse_{model_name}.pt"))

        with open(os.path.join(model_dir, "results", "mse.json"), "r") as f:
            info = json.load(f)

            min_rmse = float(info["best_rmse"].strip("tensor()"))
            max_pearson = float(info["best_pearson"].strip("tensor()"))
            min_loss = float(info["test_loss"].strip("tensor()"))
            max_r2 = float(info["best_r2"].strip("tensor()"))

    else:
        min_loss, min_rmse, max_pearson, max_r2 = 100, 1000, 0, 0

    # Define optimizer
    optimizer = OPTIMIZER_FACTORY[params.get("optimizer", "Adam")](
        model.parameters(), lr=params.get("lr", 0.01)
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params.update({"number_of_parameters": num_params})
    print(f"Number of parameters {num_params}")

    # Overwrite params.json file with updated parameters.
    with open(os.path.join(model_dir, "model_params.json"), "w") as fp:
        json.dump(params, fp)

    # Start training
    print("Training about to start...\n")
    t = time()

    model.save(save_top_model.format("epoch", "0", model_name))

    for epoch in range(params["epochs"]):

        model.train()
        print(params_filepath.split("/")[-1])
        print(f"== Epoch [{epoch}/{params['epochs']}] ==")
        train_loss = 0

        for ind, (smiles, gep, mut, y) in enumerate(train_loader):
            y_hat, pred_dict = model(
                torch.squeeze(smiles.to(device)), gep.to(device), mut.to(device))
            loss = model.loss(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-6)
            optimizer.step()
            train_loss += loss.item()

        print(
            "\t **** TRAINING ****   "
            f"Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {train_loss / len(train_loader):.5f}. "
            f"This took {time() - t:.1f} secs."
        )
        t = time()

        # Measure validation performance
        model.eval()
        with torch.no_grad():
            test_loss = 0
            # predictions = []
            # labels = []
            log_pres = []
            log_labels = []
            for ind, (smiles, gep, mut, y) in enumerate(test_loader):
                y_hat, pred_dict = model(
                    torch.squeeze(smiles.to(device)), gep.to(device), mut.to(device)
                )
                log_pre = pred_dict.get("log_micromolar_IC50")
                log_pres.append(log_pre)
                # predictions.append(y_hat)
                log_y = get_log_molar(y, ic50_max=max_value, ic50_min=min_value)
                log_labels.append(log_y)
                # labels.append(y)
                loss = model.loss(log_pre, log_y.to(device))
                test_loss += loss.item()

        # on the logIC50 scale
        predictions = torch.cat([p.cpu() for preds in log_pres for p in preds])
        labels = torch.cat([l.cpu() for label in log_labels for l in label])
        # 计算pearson相关系数
        test_pearson_a = pearsonr(torch.Tensor(predictions), torch.Tensor(labels))
        test_rmse_a = torch.sqrt(torch.mean((predictions - labels) ** 2))
        test_loss_a = test_loss / len(test_loader)
        test_r2_a = r2_score(torch.Tensor(predictions), torch.Tensor(labels))
        print(
            f"\t **** TESTING ****   Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {test_loss_a:.5f}, "
            f"Pearson: {test_pearson_a:.3f}, "
            f"RMSE: {test_rmse_a:.3f}, "
            f"R2: {test_r2_a:.3f}. "
        )

        def save(path, metric, typ, val=None):
            model.save(path.format(typ, metric, model_name))
            with open(os.path.join(model_dir, "results", metric + ".json"), "w") as f:
                json.dump(info, f)
            np.save(
                os.path.join(model_dir, "results", metric + "_preds.npy"),
                np.vstack([predictions, labels]),
            )
            if typ == "best":
                print(
                    f'\t New best performance in "{metric}"'
                    f" with value : {val:.7f} in epoch: {epoch}"
                )

        def update_info():
            return {
                "best_rmse": str(min_rmse),
                "best_pearson": str(float(max_pearson)),
                "test_loss": str(min_loss),
                "best_r2": str(float(max_r2)),
                "predictions": [float(p) for p in predictions],
            }

        if test_loss_a < min_loss:
            min_rmse = test_rmse_a
            min_loss = test_loss_a
            min_loss_pearson = test_pearson_a
            info = update_info()
            save(save_top_model, "mse", "best", min_loss)
            ep_loss = epoch
        if test_pearson_a > max_pearson:
            max_pearson = test_pearson_a
            max_pearson_loss = test_loss_a
            info = update_info()
            save(save_top_model, "pearson", "best", max_pearson)
            ep_pearson = epoch
        if test_r2_a > max_r2:
            max_r2 = test_r2_a
            info = update_info()
            save(save_top_model, "r2", "best", max_r2)
            ep_r2 = epoch
        if (epoch + 1) % params.get("save_model", 100) == 0:
            save(save_top_model, "epoch", str(epoch))
    print(
        "Overall best performances are: \n \t"
        f"Loss = {min_loss:.4f} in epoch {ep_loss} "
        f"\t (Pearson was {min_loss_pearson:4f}) \n \t"
        f"Pearson = {max_pearson:.4f} in epoch {ep_pearson} "
        f"\t (Loss was {max_pearson_loss:2f})"
        f"\t R2 = {max_r2:.4f} in epoch {ep_r2} "
    )
    save(save_top_model, "training", "done")
    print("Done with training, models saved, shutting down.")


if __name__ == "__main__":

    train_sensitivity_filepath = 'data/drug_sensitivity_MixedSet_train.csv'
    test_sensitivity_filepath = 'data/drug_sensitivity_MixedSet_test.csv'
    gep_filepath = 'data/OmicsExpressionProteinCodingGenesTPMLogp1-23Q2_Only_MEDICUS_GSVA.csv'
    mut_filepath = 'data/MUT_cardinality_analysis_of_variance_Only_MEDICUS.csv'
    smi_filepath = 'data/ccle-gdsc.smi'
    gene_filepath = 'data/MUDICUS_Omic_619_pathways.pkl'
    smiles_language_filepath = 'data/smiles_language/tokenizer_customized'
    model_path = 'result/model'
    params_filepath = 'data/params/TransMCA_Dense_GEP_MUT.json'
    training_name = 'TRANS_MCA_GEP_MUT_MEDICUS619_Clipping'
    # run the training
    main(
        train_sensitivity_filepath,
        test_sensitivity_filepath,
        gep_filepath,
        mut_filepath,
        smi_filepath,
        gene_filepath,
        smiles_language_filepath,
        model_path,
        params_filepath,
        training_name
    )
