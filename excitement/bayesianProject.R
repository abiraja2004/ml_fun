load("~/Google Drive/MATH355/Data/ssn2004.RData")
objs <- ls(ssn2004)
for (obj in objs) {
  try({
    print(obj)
    nbagame <- read.csv(paste0('~/Google Drive/MATH355/Data/', obj, '.csv'))
    if (nbagame[paste0('X', obj, '.fm')][1,] > 0) {
      nbagame['win'] <- 1
    } else nbagame['win'] <- 0
#    nbagame['homecourt'] <- 1
#    nbagame$win <- as.factor(nbagame$win)
#    nbagame$homecourt <- as.factor(nbagame$homecourt)
#    levels(nbagame$homecourt) <- c(1,0)
    write.csv(nbagame, paste0('~/Google Drive/MATH355/Data/', obj, '.csv'), row.names=F)
  })
}

first <- read.csv("~/Google Drive/MATH355/Data/0020400001.csv")
colnames(first) = c("gid","rid","r","ts","tsv","q","min","sec","gt","a","h","m","d","emt","emat","gst1","gst2","pt1","pt2","line","total","fm","ft","win","homecourt")
write.csv(first, "~/Google Drive/MATH355/Data/Master.csv", row.names=F)
master <- read.csv("~/Google Drive/MATH355/Data/Master.csv")

for (obj in objs) {
  try({
    print(obj)
    nbagame <- read.csv(paste0('~/Google Drive/MATH355/Data/', obj, '.csv'))
    master <- rbind(master, setNames(nbagame, c("gid","rid","r","ts","tsv","q","min","sec","gt","a","h","m","d","emt","emat","gst1","gst2","pt1","pt2","line","total","fm","ft","win","homecourt")))
  })
  write.csv(master, "~/Google Drive/MATH355/Data/Master.csv", row.names=F)
}

# Frequentist winning probability plot of tracy mcgrady game
mod1 = glm(win ~ m + gt + m:gt + line, data=master, family="binomial")
pr = makeFun(mod1)
ppoint <- pr(m = tmacGame['m'], gt = tmacGame['gt'], line = tmacGame['line'])
ppoint[390] <- 1
gameTime <- tmacGame['gt'][[1]]
gameTime[390] <- 2880

# Plot probability using frequentist estimates
tmacGame <- read.csv("~/Google Drive/MATH355/Data/0020400273.csv")
ppoint <- pr(m = tmacGame['m'], gt = tmacGame['gt'], line = tmacGame['line'])
ppoint[390] <- 1
gameTime <- tmacGame['gt'][[1]]
gameTime[390] <- 2880
plot(ppoint~gameTime, type='l', col='red', ylim=c(0,1.0), xlab="Game Time (seconds)", ylab="Win Probability", main="Win Probability for HOU-SAS (12/9/2004)")
abline(v=720)
abline(v=1440)
abline(v=2160)
#text(40, 0, "Q1")
text(760, 0, "Q2")
text(1480, 0, "Q3")
text(2200, 0, "Q4")
text(-40, 0, "HOU", font=2)
text(-40, 1, "SAS", font=2)
grid.circle(0.86, 0.30, r=0.015)
grid.circle(0.86, 0.30, r=0.010)
grid.circle(0.86, 0.30, r=0.005)

master <- read.csv("~/Google Drive/MATH355/Data/Master.csv")
testMaster <- master[c(1:50000),]

nbaGamesModel <- function() {
  #Data
  for(i in 1:length(y)) {
    y[i] ~ dbern(p[i])
    logit(p[i]) <- b0 + b1*x1[i] + b2*x2[i] + b3*x3[i] + b4*x2[i]*x3[i]
  }    
  # Prior 
  b0 ~ dnorm(0, 10)
  b1 ~ dnorm(0, 10)
  b2 ~ dnorm(0, 10)
  b3 ~ dnorm(0, 10)
  b4 ~ dnorm(0, 10)
}

nbaGamesTextModel = "model{
  #Data
  for(i in 1:length(y)) {
    y[i] ~ dbern(p[i])
    logit(p[i]) <- b0 + b1*x1[i] + b2*x2[i] + b3*x3[i] + b4*x2[i]*x3[i]
  }    
  # Prior 
  b0 ~ dnorm(0, 10)
  b1 ~ dnorm(0, 10)
  b2 ~ dnorm(0, 10)
  b3 ~ dnorm(0, 10)
  b4 ~ dnorm(0, 10)
}"

library(rjags)
library(dclone)

jdata <- list(y=master$win, x1=master$m, x2=master$gt, x3=master$line)
#jdata <- list(y=testMaster$win, x1=testMaster$m, x2=testMaster$gt, x3=testMaster$line)
jpara <- c('b0', 'b1', 'b2', 'b3', 'b4')
mod <- jags.fit(jdata, jpara, nbaGamesModel, n.chains=3, n.adapt=1000,n.update=1000,n.iter=10000)
nbaGamesJAGS = jags.model(textConnection(nbaGamesTextModel), data=jdata)
nbaGamesSim = coda.samples(nbaGamesJAGS, jpara, n.iter=10000)
nbaGamesSamples = data.frame(nbaGamesSim[[1]])
summary(nbaGamesSim)

# For summary statistics of variables game time, point difference, and vegas line
a = c(0, -46, -18)
b = c(743, -4, -8)
c = c(1471, 1, -4)
d = c(1481, 1.702, -3.415)
e = c(2202, 7, 1.5)
f = c(3780, 44, 14)
df = data.frame(a,b,c,d,e,f)
colnames(df) = c("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max.")
rownames(df) = c("Game Time", "Point Difference", "Vegas Line")

# Bayesian winning probability plot of tracy mcgrady game
bayesianFunction <- makeFun(exp(-0.8258*10^-2 + 0.1623*pointDiff + -5.311*10^-6*gameTime + -0.1776*vegasLine + 2.889*10^-5*pointDiff*gameTime)/(1+exp(-0.8258*10^-2 + 0.1623*pointDiff + -5.311*10^-6*gameTime + -0.1776*vegasLine + 2.889*10^-5*pointDiff*gameTime)) ~ pointDiff & gameTime & vegasLine)
bayesP <- bayesianFunction(pointDiff = tmacGame['m'], gameTime = tmacGame['gt'], vegasLine = tmacGame['line'])
colnames(bayesP) <- "p"
bayesP['gt'] <- tmacGame['gt'][[1]]
bayesP['m'] <- tmacGame['m'][[1]]
bayesP[390,1] <- 1
bayesP[390,2] <- 2880
bayesP[390,3] <- 1
ggplot(bayesP, aes(x=gt, y=p)) + geom_line(aes(colour=m)) + scale_colour_gradient(low="blue",high="red") + labs(title="Win Probability for HOU-SAS 12/9/2004", x="Game Time", y="Winning Probability", colour="Point Diff")

# Function to convert games into dataframes with predictions and confidence intervals for ease of plottin
nbaGameToProbabilities = function(game) {
  len = length(game[,1])
  lwr = rep(0, len)
  upr = rep(0, len)
  p = rep(0, len)
  gt = game['gt'][[1]]
  m = game['m'][[1]]
  lwr[len + 1] = game['win'][1,]
  upr[len + 1] = game['win'][1,]
  p[len + 1] = game['win'][1,]
  gt[len + 1] = gt[len]
  m[len + 1] = m[len]
    
  for(i in 1:len) {
    credibleInterval = nbaGamesCredibleInterval(game['m'][i,], game['gt'][i,], game['line'][i,])
    lwr[i] = credibleInterval[1]
    upr[i] = credibleInterval[3]
    p[i] = credibleInterval[2]
  }
  
  bayesP = data.frame(gt, m, lwr, p, upr)
  colnames(bayesP) <- c("gt", "m", "lwr", "p", "upr")
  return(bayesP)
}

nbaGamesCredibleInterval = function(pointDiff, gameTime, vegasLine) {
  predictions = exp(nbaGamesSamples[['b0']] + nbaGamesSamples[['b1']]*pointDiff + nbaGamesSamples[['b2']]*gameTime + nbaGamesSamples[['b3']]*vegasLine + nbaGamesSamples[['b4']]*pointDiff*gameTime)/(1+exp(nbaGamesSamples[['b0']] + nbaGamesSamples[['b1']]*pointDiff + nbaGamesSamples[['b2']]*gameTime + nbaGamesSamples[['b3']]*vegasLine + nbaGamesSamples[['b4']]*pointDiff*gameTime))
  lwr = quantile(predictions, 0.025)
  upr = quantile(predictions, 0.975)
  p = quantile(predictions, 0.5)
  credibleInterval = c(lwr, p, upr)
  return(credibleInterval)
}

# Plot with confidence intervals at -0.03 and +0.03
ggplot(tmacGamePredictions, aes(x=gt, y=p)) + geom_line(aes(colour=m)) + geom_ribbon(aes(ymin=lwr-0.03, ymax=upr+0.03), alpha=0.3) + scale_colour_gradient(low="blue",high="red") + labs(title="Win Probability for HOU-SAS 12/9/2004", x="Game Time", y="Winning Probability", colour="Point Diff")

# Problem: Confidence interval is very narrow
# Checking if confidence interval exists
# http://stackoverflow.com/questions/3777174/plotting-two-variables-as-lines-using-ggplot2-on-the-same-graph
library(reshape2)
testTmacGamePredictions = melt(tmacGamePredictions[,c(1,3,4,5)], id="gt")
ggplot(testTmacGamePredictions, aes(x=gt, y=value, group=variable, colour=variable)) + geom_line()

# For numerical summaries of posterior
e = c(-0.10666, 0.1603, -2.149*10^-5, -0.1824, 2.615*10^-5)
f = c(-0.08723, 0.1619, -8.232*10^-6, -0.1784, 2.84*10^-5)
g = c(-0.08257, 0.1623, -5.383*10^-6, -0.1775, 2.888*10^-5)
h = c(-0.08258, 0.1623, -5.311*10^-6, -0.1776, 2.889*10^-5)
i = c(-0.07772, 0.1626, -2.36*10^-6, -0.1767, 2.939*10^-5)
j = c(-0.05669, 0.1644, 9.341*10^-6, -0.1730, 3.139*10^-5)
postdf = data.frame(e, f, g, h, i ,j)
colnames(postdf) <- c("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max.")
rownames(postdf) <- c("beta0", "beta1", "beta2", "beta3", "beta4")

# Edit running.plot function to only plot from Alicia's link source("http://www.macalester.edu/.../BayesianFunctions.R")
running.plot = function(x,bars=FALSE,title="Running Estimate", ylab="estimate"){
  n = length(x)
  
  #Calculate the running mean of x:
  run.mean = cumsum(x)/c(1:n)
  
  #Calculate the running margin of error
  moe.run = rep(0,n-1)
  for(i in 2:(n-1)){
    moe.run[i-1] = 1.96*sd(x[1:i])/sqrt(i)
  }
  
  if(bars=="FALSE"){
    #Plot the running mean versus sample size:
    plot(c(1:n), run.mean, xlab="sample size", ylab=ylab, 
         type="l", main=title) 
  }
  if(bars=="TRUE"){
    #Plot the running mean versus sample size:
    plot(c(1:n), run.mean, xlab="sample size", ylab=ylab, 
         main="Running Sample Mean", type="l", ylim=c(min(run.mean[-1]-moe.run),max(run.mean[-1]+moe.run))) 
    lines(c(2:n), run.mean[-1] + moe.run, col=2)
    lines(c(2:n), run.mean[-1] - moe.run, col=2)
  }
}

lakersGame <- read.csv("~/Google Drive/MATH355/Data/0020400115.csv")
colnames(lakersGame) = c("gid","rid","r","ts","tsv","q","min","sec","gt","a","h","m","d","emt","emat","gst1","gst2","pt1","pt2","line","total","fm","ft","win")
write.csv(lakersGame, "~/Google Drive/MATH355/Data/lakersGame.csv", row.names=F)
lakersGamePredictions <- nbaGameToProbabilities(lakersGame)
ggplot(lakersGamePredictions, aes(x=gt, y=p)) + geom_line(aes(colour=m)) + geom_ribbon(aes(ymin=lwr, ymax=upr), alpha=0.3) + scale_colour_gradient(low="blue",high="red") + labs(x="Game Time", y="Winning Probability", colour="Point Diff") + scale_y_continuous(limits=c(0,1))

# Save RData file of R environment with objects, data, functions related to project
save.image("~/Google Drive/MATH355/finalReport.RData")
load("~/Google Drive/MATH355/finalReport.RData")

