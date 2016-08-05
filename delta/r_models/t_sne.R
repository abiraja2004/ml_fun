library(ggplot2)
library(readr)
library(Rtsne)
library(randomForest) #need the imputation functions
library(caret)
library(lubridate)

set.seed(4816)

##DEMO.  Run tSNE on the iris data set
iris_unique <- unique(iris) 
tsne_out <- Rtsne(as.matrix(iris_unique[,1:4])) # Run TSNE
plot(tsne_out$Y,col=iris$Species) # Plot the result



delta_cluster <- read.csv("C:/Users/josephdean/Desktop/kag/delta/delta_cluster.csv")
numeric_features <- delta_cluster[,c(seq(12,38))] #, seq(48,81)
numeric_features<- na.roughfix(numeric_features)
numeric_features <- data.frame(scale(numeric_features))


tsne <- Rtsne(as.matrix(numeric_features), check_duplicates = FALSE, pca = TRUE, 
              perplexity=30, theta=0.5, dims=2)


embedding <- as.data.frame(tsne$Y)
embed<- cbind(embedding, delta_cluster)

survey <- delta_cluster[,seq(3,5)]
survey<- na.roughfix(survey)

survey_tsne <-Rtsne(as.matrix(survey), check_duplicates = FALSE, pca = FALSE, 
                    perplexity=30, theta=0.5, dims=2)


write_csv(embed, "c:/users/josephdean/desktop/kag/delta/embedded.csv")

s<-as.data.frame(survey_tsne$Y)
s<-cbind(s, survey)
ggplot(s, aes(x=V1, y=V2, color=gts_1, alpha=.9)) + geom_point()

embedding_otto$target <- train_otto$target
e2<-cbind(embedding, numeric_features)
e2$total_tkt_rev_yr1 <- train$total_tkt_rev_yr1


ggplot(embed, aes(x=V1, y=V2, color=as.factor(CLUSTER), alpha=.3)) +
  geom_point(size=1) +
  xlab("") + ylab("") +
  theme_light(base_size=20)




