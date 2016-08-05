# load libraries
library(h2o)
library(stringr)
library(readr)
library(ggplot2)
library(reshape)


# set instance
#h2oServer = h2o.init(ip="lyuba01", port = 54321, nthreads=-1, startH2O=F) # remote
h2oServer = h2o.init(nthreads=-1, max_mem_size='8g', startH2O=T) # local

# load data
path_train = normalizePath("../data/train.zip")
path_test = normalizePath("../data/test.zip")
train_hex = h2o.importFile(h2oServer, path = path_train, key="train.hex")
test_hex = h2o.importFile(h2oServer, path = path_test, key="test.hex")



# Read competition data files:
train <- read <- csv("../input/train.csv")
#names(train)
d <- melt(train[,-c(1:5)])
ggplot(d,aes(x = value)) +
      facet <- wrap(~variable,scales = "free_x") +
      geom <- histogram()





# reduce dimensions
vars = colnames(train_hex)
predictors = c(2:34)
targets = vars[35:38]

# model settings
ensemble_size = 20
n_fold = 20
reproducible_mode = F
seed0 = 1337 # ignored if above is F

# score helpers
MSEs = matrix(0, nrow = 1, ncol = length(targets))
RMSEs = matrix(0, nrow = 1, ncol = length(targets))
CMRMSE = 0

# main
for (resp in 1:length(targets)) {
  cat("Training a DNN model for", targets[resp], "\n")
  
  # grid search with n-fold cross-validation
  cvmodel =
    h2o.deeplearning(x = predictors,
                     y = targets[resp],
                     data = train_hex,
#                     validation = test_hex,
                     nfolds = n_fold,
                     classification = T,
		     autoencoder = F,
                     activation="RectifierWithDropout",
                     hidden = c(100,100,100),
                     hidden_dropout_ratios = c(0.0,0.0,0.0),
                     input_dropout_ratio = 0,
                     epochs = 100,
                     l1 = c(0,1e-5),
                     l2 = c(0,1e-5), 
                     rho = 0.99, 
                     epsilon = 1e-8, 
                     train_samples_per_iteration = -2,
                     reproducible = reproducible_mode,
#                     loss = 
                     seed = seed0 + resp
                     )

  # collect cross-validation error
  cvmodel = cvmodel@model[[1]] # if cvmodel is a grid search model
  MSE = cvmodel@model$valid_sqr_error
  RMSE = sqrt(MSE)
  CMRMSE = CMRMSE + RMSE # column-mean-RMSE
  MSEs[resp] = MSE
  RMSEs[resp] = RMSE
  cat("\nCross-validated MSEs so far:", MSEs)
  cat("\nCross-validated RMSEs so far:", RMSEs)
  cat("\nCross-validated CMRMSE so far:", CMRMSE/resp)
  cat("\nTaking parameters from grid search winner for", targets[resp], "...\n")
  p = cvmodel@model$params

  # build ensemble model on full training data
  for (n in 1:ensemble_size) {
    cat("Building ensemble model", n, "of", ensemble_size, "for", targets[resp], "...\n")
    model =
      h2o.deeplearning(x = predictors,
                       y = targets[resp],
                       key = paste0(targets[resp], "_cv_ensemble_", n, "_of_", ensemble_size),
                       data = train_hex, 
                       classification = F,
                       activation = p$activation,
                       hidden = p$hidden,
                       hidden_dropout_ratios = p$hidden_dropout_ratios,
                       input_dropout_ratio = p$input_dropout_ratio,
                       epochs = p$epochs,
                       l1 = p$l1,
                       l2 = p$l2,
                       rho = p$rho,
                       epsilon = p$epsilon,
                       train_samples_per_iteration = p$train_samples_per_iteration,
                       reproducible = p$reproducible,
                       seed = p$seed + n
                       )
    
    # aggregate ensemble model predictions
    test_preds = h2o.predict(model, test_hex)
    if (n == 1) {
      test_preds_blend = test_preds
    } else {
      test_preds_blend = cbind(test_preds_blend, test_preds[,1])
    }
  }
  
  # output
  cat (paste0("\nNumber of ensemble models: ", ncol(test_preds_blend)))
  ensemble_average = matrix("ensemble_average", nrow = nrow(test_preds_blend), ncol = 1)
  ensemble_average = rowMeans(as.data.frame(test_preds_blend)) # Simple ensemble average, consider blending/stacking
  ensemble_average = as.data.frame(ensemble_average)
  
  colnames(ensemble_average)[1] = targets[resp]
  if (resp == 1) {
      final_submission = cbind(as.data.frame(test_hex[,1]), ensemble_average)
  } else {
      final_submission = cbind(final_submission, ensemble_average)
  }
  print(head(final_submission))
  
  # clear old models and KV store to clean mem
  ls_temp = h2o.ls(h2oServer)
  for (n_ls in 1:nrow(ls_temp)) {
      if (str_detect(ls_temp[n_ls, 1], "DeepLearning")) {
          h2o.rm(h2oServer, keys = as.character(ls_temp[n_ls, 1]))
      } else if (str_detect(ls_temp[n_ls, 1], "Last.value")) {
          h2o.rm(h2oServer, keys = as.character(ls_temp[n_ls, 1]))
      }
  }
}
cat(paste0("\nOverall cross-validated CMRMSE = " , CMRMSE/length(targets)))

# write to file
path_output = normalizePath("results/ensemble_preds.csv")
write.csv(final_submission, file = path_output, quote = F, row.names=F)
