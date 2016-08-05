# load libs
library(h2o)
localH2O <- h2o.init(max_mem_size='8g', nthreads=-1)

# set paths to data
path_cloud <- getwd()
path_data <- paste0(path_cloud, "/data/orig_sample.csv")

# load data and summarize
delta <- h2o.importFile(localH2O, path = path_data, key="", parse=T, sep =",")
#summary(delta)
system.time(summary(delta))

# split data
data_split <- h2o.splitFrame(delta, ratios=0.8, shuffle=T)

# binomial glm
y = "gts_1"
#X = c("Year", "Month", "DayofMonth", "DayOfWeek", "CRSDepTime", "UniqueCarrier", "Origin", "Dest", "Distance")
delta.glm <- h2o.glm(y = y,
                     x = 1:34,
                     data = delta,
                     family = "gaussian",
                     nfolds = 0,
                     alpha = 0.5,
                     lambda_search=F,
                     use_all_factor_levels=F,
                     variable_importances=F,
                     standardize=T,
                     higher_accuracy=F)

# summarize model
system.time(delta.glm)
coefs <- delta.glm@model$coefficients
head(coefs[order(abs(coefs), decreasing=T)], 20)

## # generate pca
## delta.pca <- h2o.prcomp(delta, tol=0, max_pc=2, standardize=T)

## # summarize pca
## system.time(delta.glm)
## print(delta.pca)
## #coefs <- delta.pca@model$coefficients
## #head(coefs[order(abs(coefs), decreasing=T)], 20)
