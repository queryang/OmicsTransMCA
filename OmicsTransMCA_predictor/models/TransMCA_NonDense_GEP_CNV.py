import logging
import sys
from collections import OrderedDict

import pytoda
import torch
import torch.nn as nn
from pytoda.smiles.transforms import AugmentTensor

from OmicsTransMCA_predictor.utils.hyperparams import LOSS_FN_FACTORY, ACTIVATION_FN_FACTORY
from OmicsTransMCA_predictor.utils.interpret import monte_carlo_dropout, test_time_augmentation
from OmicsTransMCA_predictor.utils.layers import convolutional_layer, ContextAttentionLayer, dense_layer
from OmicsTransMCA_predictor.utils.utils import get_device, get_log_molar

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
    五通道；三组学；后期特征融合
"""
# TODO: 五通道组学特征不降维导致模型过大，极有可能出现过拟合；这部分模型可能需要丢弃


class TransMCA_NonDense_GEP_CNV(nn.Module):
    """Based on the MCA model in Molecular Pharmaceutics:
        https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520.
        Updates:
            - Context instead of self attention on omic data
    """

    def __init__(self, params, *args, **kwargs):
        """Constructor.

        Args:
            params (dict): A dictionary containing the parameter to built the
                dense encoder.
                TODO params should become actual arguments (use **params).

        Items in params:
            smiles_padding_length (int): Padding length for SMILES.
            smiles_embedding_size (int): dimension of tokens' embedding.
            smiles_vocabulary_size (int): size of the tokens vocabulary.
            activation_fn (string, optional): Activation function used in all
                layers for specification in ACTIVATION_FN_FACTORY.
                Defaults to 'relu'.
            batch_norm (bool, optional): Whether batch normalization is
                applied. Defaults to True.
            dropout (float, optional): Dropout probability in all
                except parametric layer. Defaults to 0.5.
            filters (list[int], optional): Numbers of filters to learn per
                SMILES convolutional layer. Defaults to [64, 64, 64].
            kernel_sizes (list[list[int]], optional): Sizes of kernels per
                SMILES convolutional layer. Defaults to  [
                    [3, params['smiles_embedding_size']],
                    [5, params['smiles_embedding_size']],
                    [11, params['smiles_embedding_size']]
                ]
                NOTE: The kernel sizes should match the dimensionality of the
                smiles_embedding_size, so if the latter is 8, the images are
                t x 8, then treat the 8 embedding dimensions like channels
                in an RGB image.
            molecule_heads (list[int], optional): Amount of attentive molecule_heads
                per SMILES embedding. Should have len(filters)+1.
                Defaults to [4, 4, 4, 4].
            stacked_dense_hidden_sizes (list[int], optional): Sizes of the
                hidden dense layers. Defaults to [1024, 512].
            smiles_attention_size (int, optional): size of the attentive layer
                for the smiles sequence. Defaults to 64.
    """

        super(TransMCA_NonDense_GEP_CNV, self).__init__(*args, **kwargs)

        # Model Parameter
        self.device = get_device()
        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get('loss_fn', 'mse')]
        self.min_max_scaling = True if params.get(
            'drug_sensitivity_processing_parameters', {}
        ) != {} else False
        if self.min_max_scaling:
            self.IC50_max = params[
                'drug_sensitivity_processing_parameters'
            ]['parameters']['max']  # yapf: disable
            self.IC50_min = params[
                'drug_sensitivity_processing_parameters'
            ]['parameters']['min']  # yapf: disable

        # Model inputs
        self.smiles_padding_length = params['smiles_padding_length']
        self.number_of_genes = params.get('number_of_genes', 619)
        self.gep_features = self.number_of_genes
        self.cnv_features = self.number_of_genes
        self.smiles_attention_size = params.get('smiles_attention_size', 64)
        self.gene_attention_size = params.get('gene_attention_size', 1)
        self.molecule_temperature = params.get('molecule_temperature', 1.)
        self.gene_temperature = params.get('gene_temperature', 1.)

        # Model architecture (hyperparameter)
        self.molecule_gep_heads = params.get('molecule_gep_heads', [2, 2, 2, 2])
        self.molecule_cnv_heads = params.get('molecule_cnv_heads', [2, 2, 2, 2])
        self.gene_heads = params.get('gene_heads', [1, 1, 1, 1])
        self.cnv_heads = params.get('cnv_heads', [1, 1, 1, 1])
        self.n_heads = params.get('n_heads', 1)
        self.num_layers = params.get('num_layers', 2)
        self.omics_dense_size = params.get('omics_dense_size', 128)
        self.filters = params.get('filters', [64, 64, 64])
        self.kernel_sizes = params.get(
            'kernel_sizes', [
                [3, params['smiles_embedding_size']],
                [5, params['smiles_embedding_size']],
                [11, params['smiles_embedding_size']]
            ]
        )
        if len(self.filters) != len(self.kernel_sizes):
            raise ValueError(
                'Length of filter and kernel size lists do not match.'
            )
        if len(self.filters) + 2 != len(self.molecule_gep_heads):
            raise ValueError(
                'Length of filter and multihead lists do not match'
            )
        self.hidden_sizes = (
            [
                # 原尺度smiles
                self.molecule_gep_heads[0] * params['smiles_embedding_size'] +
                self.molecule_cnv_heads[0] * params['smiles_embedding_size'] +
                # transformer编码后的smiles 尺度
                self.molecule_gep_heads[1] * params['smiles_embedding_size'] +
                self.molecule_cnv_heads[1] * params['smiles_embedding_size'] +
                # omics
                sum(self.gene_heads) * self.gep_features +
                sum(self.cnv_heads) * self.cnv_features +
                # 多尺度卷积
                sum(
                    [
                        h * f
                        for h, f in zip(self.molecule_gep_heads[1:], self.filters)
                    ]
                ) * 2   # 组学数据数量
            ] + params.get('stacked_dense_hidden_sizes', [1024, 512])
        )

        self.dropout = params.get('dropout', 0.5)
        self.temperature = params.get('temperature', 1.)
        self.act_fn = ACTIVATION_FN_FACTORY[
            params.get('activation_fn', 'relu')]

        # Build the model
        self.smiles_embedding = nn.Embedding(
            self.params['smiles_vocabulary_size'],
            self.params['smiles_embedding_size'],
            scale_grad_by_freq=params.get('embed_scale_grad', False)
        )

        # Transformer Encoder
        encoder = nn.TransformerEncoderLayer(d_model=self.params['smiles_embedding_size'], nhead=self.n_heads, dropout=self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder, self.num_layers)

        self.convolutional_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'convolutional_{index}',
                        convolutional_layer(
                            num_kernel,
                            kernel_size,
                            act_fn=self.act_fn,
                            batch_norm=params.get('batch_norm', False),
                            dropout=self.dropout
                        ).to(self.device)
                    ) for index, (num_kernel, kernel_size) in
                    enumerate(zip(self.filters, self.kernel_sizes))
                ]
            )
        )

        smiles_hidden_sizes = ([params['smiles_embedding_size']] +
                               [params['smiles_embedding_size']] + self.filters)

        self.molecule_attention_layers_gep = nn.Sequential(OrderedDict([
            (
                f'molecule_gep_attention_{layer}_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=smiles_hidden_sizes[layer],
                    reference_sequence_length=self.smiles_padding_length,
                    context_hidden_size=1,
                    context_sequence_length=self.number_of_genes,
                    attention_size=self.smiles_attention_size,
                    individual_nonlinearity=params.get(
                        'context_nonlinearity', nn.Sequential()
                    ),
                    temperature=self.molecule_temperature
                )
            ) for layer in range(len(self.molecule_gep_heads))
            for head in range(self.molecule_gep_heads[layer])
        ]))  # yapf: disable

        self.molecule_attention_layers_cnv = nn.Sequential(OrderedDict([
            (
                f'molecule_cnv_attention_{layer}_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=smiles_hidden_sizes[layer],
                    reference_sequence_length=self.smiles_padding_length,
                    context_hidden_size=1,
                    context_sequence_length=self.number_of_genes,
                    attention_size=self.smiles_attention_size,
                    individual_nonlinearity=params.get(
                        'context_nonlinearity', nn.Sequential()
                    ),
                    temperature=self.molecule_temperature
                )
            ) for layer in range(len(self.molecule_cnv_heads))
            for head in range(self.molecule_cnv_heads[layer])
        ]))

        # Gene attention stream
        self.gene_attention_layers = nn.Sequential(OrderedDict([
            (
                f'gene_attention_{layer}_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=1,
                    reference_sequence_length=self.number_of_genes,
                    context_hidden_size=smiles_hidden_sizes[layer],
                    context_sequence_length=self.smiles_padding_length,
                    attention_size=self.gene_attention_size,
                    individual_nonlinearity=params.get(
                        'context_nonlinearity', nn.Sequential()
                    ),
                    temperature=self.gene_temperature
                )
            ) for layer in range(len(self.molecule_gep_heads))
            for head in range(self.gene_heads[layer])
        ]))  # yapf: disable

        # CNV attention stream
        self.cnv_attention_layers = nn.Sequential(OrderedDict([
            (
                f'cnv_attention_{layer}_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=1,
                    reference_sequence_length=self.number_of_genes,
                    context_hidden_size=smiles_hidden_sizes[layer],
                    context_sequence_length=self.smiles_padding_length,
                    attention_size=self.gene_attention_size,
                    individual_nonlinearity=params.get(
                        'context_nonlinearity', nn.Sequential()
                    ),
                    temperature=self.gene_temperature
                )
            ) for layer in range(len(self.molecule_cnv_heads))
            for head in range(self.cnv_heads[layer])
        ]))

        # Only applied if params['batch_norm'] = True
        self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])
        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dense_{}'.format(ind),
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=params.get('batch_norm', True)
                        ).to(self.device)
                    ) for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        self.final_dense = (
            nn.Linear(self.hidden_sizes[-1], 1)
            if not params.get('final_activation', False) else nn.Sequential(
                OrderedDict(
                    [
                        ('projection', nn.Linear(self.hidden_sizes[-1], 1)),
                        ('sigmoidal', ACTIVATION_FN_FACTORY['sigmoid'])
                    ]
                )
            )
        )

    def forward(self, smiles, gep, cnv):
        """Forward pass through the PaccMannV2.

        Args:
            smiles (torch.Tensor): of type int and shape: [bs, smiles_padding_length]
            gep (torch.Tensor): of type float and shape: [bs, number_of_genes]
            cnv (torch.Tensor): of type float and shape: [bs, number_of_genes]

        Returns:
            (torch.Tensor, dict): predictions, prediction_dict
            predictions is IC50 drug sensitivity prediction of shape `[bs, 1]`.
            prediction_dict includes the prediction and attention weights.
        """

        gep = torch.unsqueeze(gep, dim=-1)
        cnv = torch.unsqueeze(cnv, dim=-1)
        embedded_smiles = self.smiles_embedding(smiles.to(dtype=torch.int64))

        # Transformer Encoder
        trans_smiles = self.transformer_encoder(embedded_smiles)

        # SMILES Convolutions. Unsqueeze has shape bs x 1 x T x H.
        encoded_smiles = [embedded_smiles] + [trans_smiles] + [
            self.convolutional_layers[ind]
            (torch.unsqueeze(embedded_smiles, 1)).permute(0, 2, 1)
            for ind in range(len(self.convolutional_layers))
        ]

        # Molecule context attention
        (encodings, smiles_alphas_gep, smiles_alphas_cnv,
         smiles_alphas_mut, gene_alphas, cnv_alphas, mut_alphas) = [], [], [], [], [], [], []
        for layer in range(len(self.molecule_gep_heads)):
            for head in range(self.molecule_gep_heads[layer]):
                ind = self.molecule_gep_heads[0] * layer + head
                e, a = self.molecule_attention_layers_gep[ind](
                    encoded_smiles[layer], gep
                )
                encodings.append(e)
                smiles_alphas_gep.append(a)

        for layer in range(len(self.molecule_cnv_heads)):
            for head in range(self.molecule_cnv_heads[layer]):
                ind = self.molecule_cnv_heads[0] * layer + head
                e, a = self.molecule_attention_layers_cnv[ind](
                    encoded_smiles[layer], cnv
                )
                encodings.append(e)
                # smiles_alphas_cnv.append(a)

        # Gene context attention
        for layer in range(len(self.gene_heads)):
            for head in range(self.gene_heads[layer]):
                ind = self.gene_heads[0] * layer + head

                e, a = self.gene_attention_layers[ind](
                    gep, encoded_smiles[layer], average_seq=False
                )
                encodings.append(e)
                gene_alphas.append(a)

        for layer in range(len(self.cnv_heads)):
            for head in range(self.cnv_heads[layer]):
                ind = self.cnv_heads[0] * layer + head

                e, a = self.cnv_attention_layers[ind](
                    cnv, encoded_smiles[layer], average_seq=False
                )
                encodings.append(e)
                # cnv_alphas.append(a)

        encodings = torch.cat(encodings, dim=1)

        # Apply batch normalization if specified
        inputs = self.batch_norm(encodings) if self.params.get(
            'batch_norm', False
        ) else encodings
        # NOTE: stacking dense layers as a bottleneck
        for dl in self.dense_layers:
            inputs = dl(inputs)

        predictions = self.final_dense(inputs)
        prediction_dict = {}

        if not self.training:
            # The below is to ease postprocessing
            smiles_attention = torch.cat(
                [torch.unsqueeze(p, -1) for p in smiles_alphas_gep], dim=-1
            )
            gene_attention = torch.cat(
                [torch.unsqueeze(p, -1) for p in gene_alphas], dim=-1
            )
            prediction_dict.update({
                'gene_attention': gene_attention,
                'smiles_attention': smiles_attention,
                'IC50': predictions,
                'log_micromolar_IC50':
                    get_log_molar(
                        predictions,
                        ic50_max=self.IC50_max,
                        ic50_min=self.IC50_min
                    ) if self.min_max_scaling else predictions
            })  # yapf: disable

        return predictions, prediction_dict

    def loss(self, yhat, y):
        return self.loss_fn(yhat, y)

    def _associate_language(self, smiles_language):
        """
        Bind a SMILES language object to the model. Is only used inside the
        confidence estimation.

        Arguments:
            smiles_language {[pytoda.smiles.smiles_language.SMILESLanguage]}
            -- [A SMILES language object]

        Raises:
            TypeError:
        """
        if not isinstance(
            smiles_language, pytoda.smiles.smiles_language.SMILESLanguage
        ):
            raise TypeError(
                'Please insert a smiles language (object of type '
                'pytoda.smiles.smiles_language.SMILESLanguage). Given was '
                f'{type(smiles_language)}'
            )
        self.smiles_language = smiles_language

    def load(self, path, *args, **kwargs):
        """Load model from path."""
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path, *args, **kwargs):
        """Save model to path."""
        torch.save(self.state_dict(), path, *args, **kwargs)
