# dev date 2024/1/27 23:39
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_data = pd.read_csv('../../data/drug_sensitivity_lung_CellBlind_test&prediction.csv')

# 按照drug分组，计算每组IC50与prediction的皮尔逊相关系数
df_data.groupby('drug')[['IC50', 'prediction']].apply(lambda x: x['IC50'].corr(x['prediction']))
drugs_corr = df_data.groupby('drug')[['IC50', 'prediction']].apply(lambda x: x['IC50'].corr(x['prediction']))
drugs_corr = drugs_corr.reset_index()
drugs_corr.columns = ['drug', 'corr']
# 画图：按照相关系数排序，画出每个drug的相关系数，从大到小，前10个使用红色，后10个使用蓝色
drugs_corr = drugs_corr.sort_values(by='corr', ascending=False)
# 重置索引
drugs_corr = drugs_corr.reset_index(drop=True)
drugs_corr['color'] = 'lime'
drugs_corr.loc[drugs_corr.index < 15, 'color'] = 'red'

drugs_corr.head()
# 画图
# plt.figure(figsize=(10, 6))
plt.bar(x=drugs_corr.index, height=drugs_corr['corr'], color=drugs_corr['color'],width=1)
# 画一条Y=0的线
plt.axhline(y=0, color='black', linestyle='-')
# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Bahnschrift']
plt.rcParams['axes.unicode_minus'] = False
# 取消X轴展示
plt.xticks([])
# 规定Y轴范围
# plt.ylim(-0.1, 0.9)
# plt.xticks(drugs_corr.index, drugs_corr['drug'], rotation=90)
plt.xlabel('Drugs', fontsize=15)
plt.ylabel('correlation coefficient', fontsize=15)
plt.title('correlation coefficient of drug and prediction', fontsize=20)
# Get the current axes, creating one if necessary.
ax = plt.gca()
# Set the spines (the box) visibility
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# ax.spines['bottom'].set_linewidth(2)  # X轴线宽
ax.spines['left'].set_linewidth(2)  # Y轴线宽

plt.tight_layout()
plt.savefig('all.png', dpi=300)
plt.show()
plt.close()
#############################################################

# 取前15个数据
drugs_corr_top15 = drugs_corr.loc[drugs_corr.index < 15, :]
drugs_corr_top15.head()

# plt.figure(figsize=(10, 6))
plt.bar(x=drugs_corr_top15['drug'], height=drugs_corr_top15['corr'], color=drugs_corr_top15['color'])
# X轴 旋转45度
plt.xticks(drugs_corr_top15['drug'], rotation=70,fontsize=15)
# 规定Y轴范围
# plt.ylim(-0.1, 0.9)
# plt.xticks(drugs_corr.index, drugs_corr['drug'], rotation=90)
plt.xlabel('Drugs', fontsize=15)

plt.ylabel('correlation coefficient', fontsize=15)
plt.title('correlation coefficient of drug and prediction', fontsize=20)
plt.tight_layout()
plt.savefig('top15.png', dpi=300)

plt.show()