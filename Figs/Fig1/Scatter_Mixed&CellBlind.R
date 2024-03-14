##Loading libraries
library(caret)
library(ggpubr)
setwd('D:\\WuYang\\code space\\python\\paper code\\OmicsTransMCA\\Figs\\Fig1')
## 读取csv文件
pre_labels= read.csv("result/TransMCADense_GEP_DrugBlind_predictions&labels.csv", sep=",", header=T, stringsAsFactors = F)
#取第一列为predictions，第二列为labels
predictions=pre_labels[,1]
labels=pre_labels[,2]
##computing metrics
perf = data.frame(
  Rsquare = R2(predictions,labels),
  correlation = cor(predictions, labels)
  
)
print(R2(predictions,labels))
print(cor(predictions, labels))
##Plotting density scatter plot for actual vs predicted labels
df = cbind.data.frame(labels,predictions)
colnames(df) = c("Actual","Predicted")

get_density <- function(x, y, ...) {
  dens <- MASS::kde2d(x, y, ...)
  ix <- findInterval(x, dens$x)
  iy <- findInterval(y, dens$y)
  ii <- cbind(ix, iy)
  return(dens$z[ii])
}
df$density <- get_density(df$Actual, df$Predicted,n=50)

g=ggscatter(df, x = "Actual", y = "Predicted", 
            add = "reg.line", conf.int = TRUE, 
            cor.coef = FALSE, cor.method = "pearson",color = "density",
            xlab = "Observed GDSC LN IC50", ylab = "Predicted LN IC50",add.params = list(color="black"),cor.coef.size = 20)
g+  scale_colour_gradientn(colours = terrain.colors(10))+theme_classic() +theme(axis.text=element_text(size=20), axis.title = element_text(size = 20))



