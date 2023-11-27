# dev date 2023/11/25 14:28
from typing import Tuple

from pytoda.datasets import SMILESTokenizerDataset
from pytoda.smiles import SMILESLanguage
from torch.utils.data import Dataset
import torch
import pandas as pd


class OmicsDrugSensitivityDataset_GEP_CNV(Dataset):
    def __init__(self,
                 drug_sensitivity_filepath: str,
                 smiles_filepath: str,
                 gep_filepath: str,
                 cnv_filepath: str,
                 gep_standardize: bool = False,
                 cnv_standardize: bool = False,
                 drug_sensitivity_dtype: torch.dtype = torch.float,
                 gep_dtype: torch.dtype = torch.float,
                 cnv_dtype: torch.dtype = torch.float,
                 smiles_language: SMILESLanguage = None,
                 drug_sensitivity_min_max: bool = True,
                 column_names: Tuple[str] = ['drug', 'cell_line', 'IC50'],
                 padding: bool = True,
                 padding_length: int = None,
                 add_start_and_stop: bool = False,
                 augment: bool = False,
                 canonical: bool = False,
                 kekulize: bool = False,
                 all_bonds_explicit: bool = False,
                 all_hs_explicit: bool = False,
                 randomize: bool = False,
                 remove_bonddir: bool = False,
                 remove_chirality: bool = False,
                 selfies: bool = False,
                 sanitize: bool = True,
                 vocab_file: str = None,
                 iterate_dataset: bool = True,
                 ):
        self.drug_sensitivity = pd.read_csv(drug_sensitivity_filepath, index_col=0)
        self.smiles = pd.read_csv(smiles_filepath,header=None)
        self.gep_standardize = gep_standardize
        self.cnv_standardize = cnv_standardize
        self.drug_sensitivity_dtype = drug_sensitivity_dtype
        self.gep_dtype = gep_dtype
        self.cnv_dtype = cnv_dtype
        self.smiles_language = smiles_language
        self.drug_sensitivity_min_max = drug_sensitivity_min_max
        self.drug_sensitivity_processing_parameters = {}
        self.column_names = column_names
        self.drug_name, self.cell_name, self.label_name = self.column_names
        if gep_filepath is not None:
            self.gep = pd.read_csv(gep_filepath, index_col=0)
        if cnv_filepath is not None:
            self.cnv = pd.read_csv(cnv_filepath, index_col=0)

        if gep_standardize:
            # TODO: implement
            pass
        if cnv_standardize:
            # TODO: implement
            pass


        # SMILES
        self.smiles_dataset = SMILESTokenizerDataset(
            smiles_filepath,
            smiles_language=smiles_language,
            augment=augment,
            canonical=canonical,
            kekulize=kekulize,
            all_bonds_explicit=all_bonds_explicit,
            all_hs_explicit=all_hs_explicit,
            remove_bonddir=remove_bonddir,
            remove_chirality=remove_chirality,
            selfies=selfies,
            sanitize=sanitize,
            randomize=randomize,
            padding=padding,
            padding_length=padding_length,
            add_start_and_stop=add_start_and_stop,
            vocab_file=vocab_file,
            iterate_dataset=iterate_dataset,
        )

        # filter data based on the availability
        drug_mask = self.drug_sensitivity[self.drug_name].isin(
            set(self.smiles_dataset.keys())
        )
        profile_mask = self.drug_sensitivity[self.cell_name].isin(
            set(self.gep.index)
        )
        self.drug_sensitivity = self.drug_sensitivity.loc[
            drug_mask & profile_mask
            ]

        # to investigate missing ids per entity
        self.masks_df = pd.concat([drug_mask, profile_mask], axis=1)
        self.masks_df.columns = [self.drug_name, self.cell_name]


        # NOTE: optional min-max scaling
        if self.drug_sensitivity_min_max:
            minimum = self.drug_sensitivity_processing_parameters.get(
                'min', self.drug_sensitivity[self.label_name].min()
            )
            maximum = self.drug_sensitivity_processing_parameters.get(
                'max', self.drug_sensitivity[self.label_name].max()
            )
            self.drug_sensitivity[self.label_name] = (
                self.drug_sensitivity[self.label_name] - minimum
            ) / (maximum - minimum)
            self.drug_sensitivity_processing_parameters = {
                'processing': 'min_max',
                'parameters': {'min': minimum, 'max': maximum},
            }

    def __len__(self):
        return len(self.drug_sensitivity)

    def __getitem__(self, index):
        # drug sensitivity
        selected_sample = self.drug_sensitivity.iloc[index]
        ic50_tensor = torch.tensor(
            [selected_sample[self.label_name]],
            dtype=self.drug_sensitivity_dtype,
        )
        # SMILES
        token_indexes_tensor = self.smiles_dataset.get_item_from_key(
            selected_sample[self.drug_name]
        )
        # omics data
        gene_expression_tensor = torch.tensor((
            self.gep.loc)[selected_sample[self.cell_name]],
            dtype=self.gep_dtype)
        cnv_tensor = torch.tensor((
            self.cnv.loc)[selected_sample[self.cell_name]],
            dtype=self.cnv_dtype)

        return (token_indexes_tensor, gene_expression_tensor,
                cnv_tensor, ic50_tensor)
