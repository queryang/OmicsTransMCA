# dev date 2024/1/26 22:59
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df_data = pd.read_csv("../../data/LUNG_Filtered_barplot.csv")

# 计算每个cell_line，prediction列和IC50列的皮尔逊相关系数和R2，保存到一个dataframe中，再加上对应的cell_line和OncotreeCode
df_data = df_data.groupby(["cell_line","OncotreeCode"]).apply(lambda x: pd.Series({
    "Pearson":x["prediction"].corr(x["IC50"],method="pearson"),
    "R2":x["prediction"].corr(x["IC50"],method="spearman")**2,
    'RMSE':((x["prediction"]-x["IC50"])**2).mean()**0.5,
})).reset_index()
#根据OncotreeCode排序
df_data = df_data.sort_values(by="OncotreeCode")


# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Bahnschrift']
plt.rcParams['axes.unicode_minus'] = False
# 设置颜色方案
# palette = sns.color_palette("muted", 5)
# 'LUAD','LCLC','NSCLC','LUSC','SCLC'
palette = {'LUAD':'#b9f2f0',
           'LCLC':'#d0bbff',
           'NSCLC':'#ff9f9b',
           'LUSC':'#66c2a5',
           'SCLC':'#a6d854'}
bar_plot = sns.barplot(x="cell_line",y="Pearson",hue="OncotreeCode",data=df_data,palette=palette,edgecolor='black')
# 设置x轴标签倾斜45度
plt.xticks(rotation=45)
bar_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Subtype')

plt.show()
