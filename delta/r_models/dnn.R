# The following two commands remove any previously installed H2O packages for R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, we download packages that H2O depends on.
if (! ("methods" %in% rownames(installed.packages()))) { install.packages("methods") }
if (! ("statmod" %in% rownames(installed.packages()))) { install.packages("statmod") }
if (! ("stats" %in% rownames(installed.packages()))) { install.packages("stats") }
if (! ("graphics" %in% rownames(installed.packages()))) { install.packages("graphics") }
if (! ("RCurl" %in% rownames(installed.packages()))) { install.packages("RCurl") }
if (! ("jsonlite" %in% rownames(installed.packages()))) { install.packages("jsonlite") }
if (! ("tools" %in% rownames(installed.packages()))) { install.packages("tools") }
if (! ("utils" %in% rownames(installed.packages()))) { install.packages("utils") }

# Now we download, install and initialize the H2O package for R.
install.packages("h2o", type="source", repos=(c("http://s3.amazonaws.com/h2o-release/h2o/master/3422/R")))
library(h2o)
localH2O = h2o.init(nthreads=-1)
h2o.clusterInfo(localH2O)

# define paths to data
path_train <- "/home/kcavagnolo/delta/data/train.zip"
path_test <- "/home/kcavagnolo/delta/data/test.zip"

# load data into cluster
train.hex <- h2o.importFile(path = path_train)
test.hex <- h2o.importFile(path = path_test)

# split data into 80:20 pieces to train and cross-validate
train_hex_split <- h2o.splitFrame(train.hex, ratios = 0.8)

# step through params for which we want predicitons
ls_label <- c("gts_1", "dl_segs_yr1", "dl_seg_rev_yr1", "total_miles_flown_yr1")
for (n_label in 1:4) {
  cat("\n\nTraining DNN model for", ls_label[n_label], "\n")
  model <- h2o.deeplearning(x = 2:38,
                            y = 3,
                            data = train_hex_split[[1]],
                            validation = train_hex_split[[2]],
                            activation = "Rectifier",
                            hidden = c(200, 200, 200),
                            epochs = 100,
                            classification = FALSE,
                            balance_classes = FALSE)
  print(model)
  raw_sub[, (n_label + 1)] <- as.matrix(h2o.predict(model, test_hex))
}

# save results to csv
path_output <- paste0(path_cloud, "/results/prediction.csv")
write.csv(raw_sub, file = path_output, row.names = FALSE)
print(sessionInfo())
print(Sys.info())
