# dev date 2023/12/30 16:40
import numpy as np
import torch

from OmicsTransMCA_predictor.utils.loss_functions import pearsonr, r2_score

# import pandas as pd
#
# df_test = pd.read_csv('../data/drug_sensitivity_CellBlind_test&prediction.csv', index_col=0)
# print(df_test.shape)
# # 取前12480个
# df_test = df_test.iloc[:12480]
# df_test = df_test['IC50']
# print(df_test.shape)

data = np.load('models/TRANS_MCA_GEP_CNV_MUT_MEDICUS619_1536_Clipping/results/mse_preds.npy')

pred = data[0]
label = data[1]
print(pred.shape)
print(label.shape)

pred = torch.Tensor(pred)
label = torch.Tensor(label)

test_pearson_a = pearsonr(pred, label)
# test_loss_a = test_loss / len(test_loader)
test_r2_a = r2_score(pred, label)

print(test_pearson_a)
print(test_r2_a)