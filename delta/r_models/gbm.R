
library(gbm) #Branch!
delta <- read.csv("C:/Users/josephdean/Desktop/kag/delta/delta_cluster.csv")
delta$random_numbers <- runif(length(delta$response_dt))
delta <- delta[delta$gts_1 != 4,]     #Remove 4 as indeterminate (no impact on NPS)
delta$detractor <- ifelse(delta$gts_1==5,0,1)  #create target variable from survey
#delta$trans_fmiles <- log(delta$F_miles_flown_yr1)  #log transform miles flown

delta2<-delta[is.na(delta$detractor)==FALSE,]  #Excluded missing responses.  Assumption is that data is missing at random.  

train<-delta2[delta2$random_numbers < .75,] 
test <- delta2[delta2$random_numbers >=.75,]

#!

##Variables performing worse than random_numbers commented out
gbm1<- gbm(formula = 
             detractor ~
      #       as.factor(CLUSTER) +  ##Doesn't predict well
        #     random_numbers + ##use random numbers for benchmarking variable importance
             PFL_FLEET_TYP_CD +
         #    comped_med+
         #    natural_med+
         #    never_comped+
          #   lapsed_med+
          #   Lesiure_tkts_yr1+
          #   Business_tkts_yr1+
             total_ap_yr1+
         #    int_tkts_yr1+
         #    dom_tkts_yr1+
         #    dl_miles_flown_yr1+
          #   dl_segs_yr1+
          #  dl_seg_rev_yr1+
           #  F_tkts_yr1+
           #  F_tkt_rev_yr1+
 ###!            F_miles_flown_yr1+
         #    C_tkts_yr1+
       #      C_tkt_rev_yr1+
        #     C_miles_flown_yr1+
          #   Y_tkts_yr1+
             Y_tkt_rev_yr1+
             Y_miles_flown_yr1+
        #     F_upsell_tkts_yr1+
        #     F_upsell_tkt_rev_yr1+
          #   F_upsell_miles_flown_yr1+
          #   total_miles_flown_yr1+
         #    total_tkts_yr1+
         #    total_tkt_rev_yr1+
             dist_partners+
       #      tenure+
        #     DSC_Entries+
        #     age+
             hhi +
        trans_fmiles , 
           distribution = "bernoulli",
           data=train,
           interaction.depth=3,
           n.minobsinnode=10,
            n.trees=6000,  
           shrinkage=.001,
           bag.fraction=.5,
           cv.folds=5,
           n.cores=2
)

summary(gbm1)
gbm.perf(gbm1)


#View marginal effects of variables.  
plot(gbm1, i=4) 
plot(gbm1, i=1) 

#View joint marginals
plot(gbm1, i=c(6,2)) 

#Evaluate the model on the validation data
test$preds <- predict(gbm1, test)
gbm.roc.area(test$detractor, test$preds)


