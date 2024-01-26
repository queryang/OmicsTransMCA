##Loading libraries
library(caret)
library(ggpubr)
setwd('D:\\WuYang\\code space\\python\\paper code\\OmicsTransMCA\\Figs\\Fig1')
## 读取csv文件
pre_labels = read.csv("../data/drug_sensitivity_CellBlind_test&prediction.csv", sep=",", header=T, stringsAsFactors = F)
# 取cell_line为KARPAS620的数据,并取IC50列和prediction列
data_KARPAS620 = pre_labels[pre_labels$cell_line=="KARPAS620",c("IC50","prediction")]
# 取cell_line为MM1S的数据,并取IC50列和prediction列
data_MM1S = pre_labels[pre_labels$cell_line=="MM1S",c("IC50","prediction")]
#data_KARPAS620取第一列为labels，第二列为predictions
data_KARPAS620_labels=data_KARPAS620[,1]
data_KARPAS620_predictions=data_KARPAS620[,2]
#data_MM1S取第一列为labels，第二列为predictions
data_MM1S_labels=data_MM1S[,1]
data_MM1S_predictions=data_MM1S[,2]

##computing metrics
perf_KARPAS620 = data.frame(
  Rsquare = R2(data_KARPAS620_predictions,data_KARPAS620_labels),
  correlation = cor(data_KARPAS620_predictions, data_KARPAS620_labels)

)
perf_MM1S = data.frame(
  Rsquare = R2(data_MM1S_predictions,data_MM1S_labels),
  correlation = cor(data_MM1S_predictions, data_MM1S_labels)

)
print(R2(data_KARPAS620_predictions,data_KARPAS620_labels))
print(cor(data_KARPAS620_predictions, data_KARPAS620_labels))
print(R2(data_MM1S_predictions,data_MM1S_labels))
print(cor(data_MM1S_predictions, data_MM1S_labels))

df_KARPAS620 = cbind.data.frame(data_KARPAS620_predictions,data_KARPAS620_labels)
colnames(df_KARPAS620) = c("Observed GDSC LN IC50","Predicted LNIC50")
df_MM1S = cbind.data.frame(data_MM1S_predictions,data_MM1S_labels)
colnames(df_MM1S) = c("Observed GDSC LN IC50","Predicted LNIC50")

group = c(rep("PCM.KARPAS620",216),rep("PCM.MM1S",215))
df = rbind.data.frame(df_KARPAS620,df_MM1S)
df1 = cbind(df,group)

ggscatter(df1, x = "Observed GDSC LN IC50", y = "Predicted LNIC50",
          add = "reg.line",
          conf.int = TRUE,cor.coef.size = 5 ,
          color = "group", palette = c("red","blue"),
          shape = "group"
)+stat_cor(aes(color = group))

