# Course Project
Rose Nyameke  
November 21, 2015  
I built my model with considerations for speed and accuracy. My first thought was
to try different models (specifically, random forests and multiple boosting methods),
and to pick the best model based on accuracy. However, due to computation limitations,
I built one model with Generalized Boosted Regression Modeling with 10-fold cross-
validation, and because I was satisfied with the accuracy, I did not build any other
models. When I used my model to predict the values of "classe" on the test data
set, I was able to obtain the correct values for all the cases, confirming the
accuracy of the model.

##Data Processing

```r
data <- read.csv("pml-training.csv")
library(plyr)
library(dplyr)
library(caret)
library(ggplot2)
library(gbm)
```

I took a simple approach to cleaning the data, and did not use any of the pre-
processing functions in the caret package. Instead, I first removed any columns
that were not numeric (except the response variable), because I noticed that these
were missing many values. I then removed columns that had NAs, after which I 
removed columns that were highly correlated with other columns (threshold of 0.75).
Lastly, I removed the timestamp variables (as they were not, strictly speaking, device
measurements), and this left me with 31 predictor variables, with which I built my model.


```r
#keeping only the numeric variables
data_wo_factors <- data[sapply(data, is.numeric)]

#re-introducing the classe variable
data_wo_factors <- cbind(data$classe, data_wo_factors)

#removing the row numbering variable
data_wo_factors <- data_wo_factors[, -2]

#removing columns with nas
data_wo_nas <- data_wo_factors[, colSums(is.na(data_wo_factors)) == 0]

#finding highly correlated predictor variables
correlationMatrix <- cor(data_wo_nas[,-1])
highcorr <- findCorrelation(correlationMatrix, cutoff = 0.75)

#removing the highly correlated columns
data_wo_corr <- data_wo_nas[,-highcorr]

#removing the timestamp columns
data_wo_time <- data_wo_corr[, -(2:3)]

#renaming the classe column
names(data_wo_time)[1] <- "classe"

clean_data <- data_wo_time
```

I built one model, using 10-fold cross-validation and generalized boosted regression
modeling.

```r
#building model using gbm and cross-fold validation (for speed reasons)
model <- train(classe ~ ., method = "gbm", trControl = trainControl(method = "cv",
               number = 10), data = clean_data, verbose = TRUE)
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1243
##      2        1.5281             nan     0.1000    0.0816
##      3        1.4731             nan     0.1000    0.0595
##      4        1.4338             nan     0.1000    0.0499
##      5        1.4007             nan     0.1000    0.0396
##      6        1.3743             nan     0.1000    0.0416
##      7        1.3476             nan     0.1000    0.0348
##      8        1.3251             nan     0.1000    0.0297
##      9        1.3060             nan     0.1000    0.0315
##     10        1.2852             nan     0.1000    0.0295
##     20        1.1477             nan     0.1000    0.0147
##     40        0.9952             nan     0.1000    0.0086
##     60        0.8989             nan     0.1000    0.0053
##     80        0.8276             nan     0.1000    0.0063
##    100        0.7682             nan     0.1000    0.0043
##    120        0.7193             nan     0.1000    0.0033
##    140        0.6794             nan     0.1000    0.0027
##    150        0.6612             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1677
##      2        1.4976             nan     0.1000    0.1172
##      3        1.4217             nan     0.1000    0.0926
##      4        1.3614             nan     0.1000    0.0748
##      5        1.3126             nan     0.1000    0.0637
##      6        1.2715             nan     0.1000    0.0525
##      7        1.2370             nan     0.1000    0.0476
##      8        1.2061             nan     0.1000    0.0521
##      9        1.1729             nan     0.1000    0.0413
##     10        1.1460             nan     0.1000    0.0377
##     20        0.9594             nan     0.1000    0.0216
##     40        0.7530             nan     0.1000    0.0093
##     60        0.6357             nan     0.1000    0.0093
##     80        0.5453             nan     0.1000    0.0055
##    100        0.4827             nan     0.1000    0.0046
##    120        0.4305             nan     0.1000    0.0026
##    140        0.3869             nan     0.1000    0.0036
##    150        0.3677             nan     0.1000    0.0033
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2028
##      2        1.4763             nan     0.1000    0.1401
##      3        1.3849             nan     0.1000    0.1182
##      4        1.3110             nan     0.1000    0.0915
##      5        1.2520             nan     0.1000    0.0853
##      6        1.1979             nan     0.1000    0.0679
##      7        1.1542             nan     0.1000    0.0604
##      8        1.1150             nan     0.1000    0.0585
##      9        1.0784             nan     0.1000    0.0492
##     10        1.0455             nan     0.1000    0.0481
##     20        0.8212             nan     0.1000    0.0320
##     40        0.5992             nan     0.1000    0.0136
##     60        0.4751             nan     0.1000    0.0063
##     80        0.3911             nan     0.1000    0.0045
##    100        0.3330             nan     0.1000    0.0047
##    120        0.2849             nan     0.1000    0.0025
##    140        0.2473             nan     0.1000    0.0031
##    150        0.2321             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1208
##      2        1.5276             nan     0.1000    0.0835
##      3        1.4730             nan     0.1000    0.0603
##      4        1.4328             nan     0.1000    0.0456
##      5        1.4018             nan     0.1000    0.0417
##      6        1.3745             nan     0.1000    0.0424
##      7        1.3473             nan     0.1000    0.0373
##      8        1.3241             nan     0.1000    0.0301
##      9        1.3047             nan     0.1000    0.0265
##     10        1.2878             nan     0.1000    0.0291
##     20        1.1471             nan     0.1000    0.0137
##     40        0.9939             nan     0.1000    0.0074
##     60        0.8972             nan     0.1000    0.0050
##     80        0.8259             nan     0.1000    0.0050
##    100        0.7672             nan     0.1000    0.0039
##    120        0.7193             nan     0.1000    0.0026
##    140        0.6777             nan     0.1000    0.0021
##    150        0.6598             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1736
##      2        1.4962             nan     0.1000    0.1173
##      3        1.4194             nan     0.1000    0.0928
##      4        1.3588             nan     0.1000    0.0742
##      5        1.3105             nan     0.1000    0.0643
##      6        1.2693             nan     0.1000    0.0567
##      7        1.2332             nan     0.1000    0.0469
##      8        1.2022             nan     0.1000    0.0471
##      9        1.1720             nan     0.1000    0.0425
##     10        1.1451             nan     0.1000    0.0351
##     20        0.9572             nan     0.1000    0.0174
##     40        0.7570             nan     0.1000    0.0099
##     60        0.6301             nan     0.1000    0.0063
##     80        0.5402             nan     0.1000    0.0057
##    100        0.4761             nan     0.1000    0.0033
##    120        0.4255             nan     0.1000    0.0031
##    140        0.3816             nan     0.1000    0.0025
##    150        0.3637             nan     0.1000    0.0024
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2098
##      2        1.4760             nan     0.1000    0.1470
##      3        1.3811             nan     0.1000    0.1142
##      4        1.3083             nan     0.1000    0.0920
##      5        1.2478             nan     0.1000    0.0761
##      6        1.1985             nan     0.1000    0.0709
##      7        1.1540             nan     0.1000    0.0622
##      8        1.1156             nan     0.1000    0.0595
##      9        1.0779             nan     0.1000    0.0564
##     10        1.0412             nan     0.1000    0.0459
##     20        0.8209             nan     0.1000    0.0242
##     40        0.6004             nan     0.1000    0.0174
##     60        0.4722             nan     0.1000    0.0072
##     80        0.3939             nan     0.1000    0.0040
##    100        0.3297             nan     0.1000    0.0039
##    120        0.2814             nan     0.1000    0.0036
##    140        0.2454             nan     0.1000    0.0019
##    150        0.2306             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1217
##      2        1.5283             nan     0.1000    0.0805
##      3        1.4743             nan     0.1000    0.0618
##      4        1.4330             nan     0.1000    0.0482
##      5        1.4007             nan     0.1000    0.0449
##      6        1.3720             nan     0.1000    0.0364
##      7        1.3479             nan     0.1000    0.0346
##      8        1.3259             nan     0.1000    0.0301
##      9        1.3065             nan     0.1000    0.0268
##     10        1.2891             nan     0.1000    0.0311
##     20        1.1519             nan     0.1000    0.0141
##     40        1.0000             nan     0.1000    0.0066
##     60        0.9037             nan     0.1000    0.0054
##     80        0.8318             nan     0.1000    0.0055
##    100        0.7731             nan     0.1000    0.0033
##    120        0.7253             nan     0.1000    0.0032
##    140        0.6835             nan     0.1000    0.0024
##    150        0.6646             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1646
##      2        1.5010             nan     0.1000    0.1170
##      3        1.4250             nan     0.1000    0.0928
##      4        1.3649             nan     0.1000    0.0748
##      5        1.3166             nan     0.1000    0.0653
##      6        1.2762             nan     0.1000    0.0581
##      7        1.2392             nan     0.1000    0.0512
##      8        1.2063             nan     0.1000    0.0423
##      9        1.1789             nan     0.1000    0.0412
##     10        1.1532             nan     0.1000    0.0468
##     20        0.9659             nan     0.1000    0.0184
##     40        0.7618             nan     0.1000    0.0089
##     60        0.6325             nan     0.1000    0.0051
##     80        0.5458             nan     0.1000    0.0041
##    100        0.4798             nan     0.1000    0.0049
##    120        0.4271             nan     0.1000    0.0024
##    140        0.3850             nan     0.1000    0.0036
##    150        0.3672             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2017
##      2        1.4779             nan     0.1000    0.1449
##      3        1.3856             nan     0.1000    0.1117
##      4        1.3129             nan     0.1000    0.0954
##      5        1.2515             nan     0.1000    0.0795
##      6        1.2012             nan     0.1000    0.0664
##      7        1.1589             nan     0.1000    0.0603
##      8        1.1200             nan     0.1000    0.0505
##      9        1.0876             nan     0.1000    0.0643
##     10        1.0470             nan     0.1000    0.0440
##     20        0.8196             nan     0.1000    0.0217
##     40        0.5999             nan     0.1000    0.0166
##     60        0.4770             nan     0.1000    0.0056
##     80        0.3953             nan     0.1000    0.0079
##    100        0.3348             nan     0.1000    0.0035
##    120        0.2857             nan     0.1000    0.0024
##    140        0.2494             nan     0.1000    0.0019
##    150        0.2352             nan     0.1000    0.0025
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1201
##      2        1.5270             nan     0.1000    0.0814
##      3        1.4722             nan     0.1000    0.0586
##      4        1.4315             nan     0.1000    0.0502
##      5        1.3982             nan     0.1000    0.0445
##      6        1.3689             nan     0.1000    0.0372
##      7        1.3450             nan     0.1000    0.0351
##      8        1.3216             nan     0.1000    0.0290
##      9        1.3030             nan     0.1000    0.0261
##     10        1.2856             nan     0.1000    0.0291
##     20        1.1464             nan     0.1000    0.0147
##     40        0.9995             nan     0.1000    0.0105
##     60        0.9009             nan     0.1000    0.0048
##     80        0.8303             nan     0.1000    0.0048
##    100        0.7690             nan     0.1000    0.0028
##    120        0.7213             nan     0.1000    0.0026
##    140        0.6811             nan     0.1000    0.0028
##    150        0.6624             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1648
##      2        1.5000             nan     0.1000    0.1136
##      3        1.4249             nan     0.1000    0.0935
##      4        1.3649             nan     0.1000    0.0714
##      5        1.3179             nan     0.1000    0.0700
##      6        1.2739             nan     0.1000    0.0563
##      7        1.2379             nan     0.1000    0.0499
##      8        1.2050             nan     0.1000    0.0436
##      9        1.1771             nan     0.1000    0.0388
##     10        1.1526             nan     0.1000    0.0410
##     20        0.9639             nan     0.1000    0.0226
##     40        0.7638             nan     0.1000    0.0105
##     60        0.6359             nan     0.1000    0.0091
##     80        0.5428             nan     0.1000    0.0065
##    100        0.4755             nan     0.1000    0.0047
##    120        0.4241             nan     0.1000    0.0039
##    140        0.3801             nan     0.1000    0.0027
##    150        0.3636             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2027
##      2        1.4757             nan     0.1000    0.1499
##      3        1.3815             nan     0.1000    0.1118
##      4        1.3095             nan     0.1000    0.0922
##      5        1.2507             nan     0.1000    0.0812
##      6        1.1995             nan     0.1000    0.0680
##      7        1.1564             nan     0.1000    0.0683
##      8        1.1114             nan     0.1000    0.0476
##      9        1.0810             nan     0.1000    0.0578
##     10        1.0444             nan     0.1000    0.0483
##     20        0.8190             nan     0.1000    0.0338
##     40        0.6010             nan     0.1000    0.0125
##     60        0.4736             nan     0.1000    0.0072
##     80        0.3905             nan     0.1000    0.0047
##    100        0.3314             nan     0.1000    0.0051
##    120        0.2857             nan     0.1000    0.0027
##    140        0.2481             nan     0.1000    0.0028
##    150        0.2321             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1209
##      2        1.5271             nan     0.1000    0.0830
##      3        1.4725             nan     0.1000    0.0603
##      4        1.4314             nan     0.1000    0.0475
##      5        1.3996             nan     0.1000    0.0408
##      6        1.3729             nan     0.1000    0.0432
##      7        1.3459             nan     0.1000    0.0360
##      8        1.3229             nan     0.1000    0.0307
##      9        1.3030             nan     0.1000    0.0288
##     10        1.2846             nan     0.1000    0.0266
##     20        1.1476             nan     0.1000    0.0154
##     40        0.9951             nan     0.1000    0.0078
##     60        0.9023             nan     0.1000    0.0065
##     80        0.8280             nan     0.1000    0.0039
##    100        0.7700             nan     0.1000    0.0040
##    120        0.7226             nan     0.1000    0.0022
##    140        0.6813             nan     0.1000    0.0027
##    150        0.6621             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1642
##      2        1.4998             nan     0.1000    0.1160
##      3        1.4245             nan     0.1000    0.0949
##      4        1.3626             nan     0.1000    0.0764
##      5        1.3131             nan     0.1000    0.0631
##      6        1.2711             nan     0.1000    0.0586
##      7        1.2336             nan     0.1000    0.0461
##      8        1.2039             nan     0.1000    0.0446
##      9        1.1750             nan     0.1000    0.0383
##     10        1.1501             nan     0.1000    0.0379
##     20        0.9584             nan     0.1000    0.0187
##     40        0.7590             nan     0.1000    0.0183
##     60        0.6297             nan     0.1000    0.0079
##     80        0.5401             nan     0.1000    0.0062
##    100        0.4792             nan     0.1000    0.0039
##    120        0.4258             nan     0.1000    0.0028
##    140        0.3833             nan     0.1000    0.0046
##    150        0.3638             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2051
##      2        1.4766             nan     0.1000    0.1519
##      3        1.3813             nan     0.1000    0.1047
##      4        1.3125             nan     0.1000    0.0983
##      5        1.2499             nan     0.1000    0.0752
##      6        1.2016             nan     0.1000    0.0718
##      7        1.1530             nan     0.1000    0.0626
##      8        1.1123             nan     0.1000    0.0529
##      9        1.0789             nan     0.1000    0.0669
##     10        1.0369             nan     0.1000    0.0545
##     20        0.8173             nan     0.1000    0.0267
##     40        0.5998             nan     0.1000    0.0078
##     60        0.4770             nan     0.1000    0.0086
##     80        0.3901             nan     0.1000    0.0055
##    100        0.3329             nan     0.1000    0.0048
##    120        0.2850             nan     0.1000    0.0027
##    140        0.2493             nan     0.1000    0.0013
##    150        0.2353             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1221
##      2        1.5276             nan     0.1000    0.0833
##      3        1.4725             nan     0.1000    0.0604
##      4        1.4321             nan     0.1000    0.0489
##      5        1.3990             nan     0.1000    0.0405
##      6        1.3720             nan     0.1000    0.0431
##      7        1.3449             nan     0.1000    0.0312
##      8        1.3240             nan     0.1000    0.0317
##      9        1.3040             nan     0.1000    0.0258
##     10        1.2872             nan     0.1000    0.0296
##     20        1.1506             nan     0.1000    0.0171
##     40        0.9981             nan     0.1000    0.0083
##     60        0.9031             nan     0.1000    0.0058
##     80        0.8308             nan     0.1000    0.0053
##    100        0.7717             nan     0.1000    0.0037
##    120        0.7225             nan     0.1000    0.0032
##    140        0.6829             nan     0.1000    0.0026
##    150        0.6651             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1648
##      2        1.5006             nan     0.1000    0.1167
##      3        1.4251             nan     0.1000    0.0895
##      4        1.3662             nan     0.1000    0.0745
##      5        1.3174             nan     0.1000    0.0650
##      6        1.2754             nan     0.1000    0.0560
##      7        1.2394             nan     0.1000    0.0488
##      8        1.2081             nan     0.1000    0.0426
##      9        1.1803             nan     0.1000    0.0471
##     10        1.1516             nan     0.1000    0.0444
##     20        0.9639             nan     0.1000    0.0244
##     40        0.7573             nan     0.1000    0.0099
##     60        0.6355             nan     0.1000    0.0062
##     80        0.5448             nan     0.1000    0.0064
##    100        0.4785             nan     0.1000    0.0035
##    120        0.4279             nan     0.1000    0.0028
##    140        0.3836             nan     0.1000    0.0034
##    150        0.3642             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2008
##      2        1.4805             nan     0.1000    0.1479
##      3        1.3864             nan     0.1000    0.1171
##      4        1.3119             nan     0.1000    0.0943
##      5        1.2513             nan     0.1000    0.0795
##      6        1.2006             nan     0.1000    0.0686
##      7        1.1567             nan     0.1000    0.0616
##      8        1.1172             nan     0.1000    0.0545
##      9        1.0828             nan     0.1000    0.0547
##     10        1.0465             nan     0.1000    0.0610
##     20        0.8217             nan     0.1000    0.0315
##     40        0.6044             nan     0.1000    0.0157
##     60        0.4799             nan     0.1000    0.0065
##     80        0.3980             nan     0.1000    0.0051
##    100        0.3340             nan     0.1000    0.0036
##    120        0.2904             nan     0.1000    0.0027
##    140        0.2522             nan     0.1000    0.0023
##    150        0.2349             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1243
##      2        1.5270             nan     0.1000    0.0812
##      3        1.4723             nan     0.1000    0.0603
##      4        1.4317             nan     0.1000    0.0472
##      5        1.4000             nan     0.1000    0.0474
##      6        1.3690             nan     0.1000    0.0359
##      7        1.3449             nan     0.1000    0.0350
##      8        1.3220             nan     0.1000    0.0293
##      9        1.3023             nan     0.1000    0.0276
##     10        1.2848             nan     0.1000    0.0270
##     20        1.1487             nan     0.1000    0.0141
##     40        0.9957             nan     0.1000    0.0087
##     60        0.8995             nan     0.1000    0.0063
##     80        0.8283             nan     0.1000    0.0030
##    100        0.7700             nan     0.1000    0.0028
##    120        0.7241             nan     0.1000    0.0027
##    140        0.6822             nan     0.1000    0.0021
##    150        0.6639             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1643
##      2        1.4985             nan     0.1000    0.1205
##      3        1.4199             nan     0.1000    0.0856
##      4        1.3646             nan     0.1000    0.0757
##      5        1.3158             nan     0.1000    0.0599
##      6        1.2773             nan     0.1000    0.0594
##      7        1.2403             nan     0.1000    0.0469
##      8        1.2088             nan     0.1000    0.0529
##      9        1.1758             nan     0.1000    0.0386
##     10        1.1513             nan     0.1000    0.0386
##     20        0.9660             nan     0.1000    0.0255
##     40        0.7542             nan     0.1000    0.0099
##     60        0.6336             nan     0.1000    0.0062
##     80        0.5450             nan     0.1000    0.0035
##    100        0.4812             nan     0.1000    0.0052
##    120        0.4281             nan     0.1000    0.0030
##    140        0.3844             nan     0.1000    0.0034
##    150        0.3640             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2053
##      2        1.4749             nan     0.1000    0.1507
##      3        1.3800             nan     0.1000    0.1090
##      4        1.3089             nan     0.1000    0.0919
##      5        1.2490             nan     0.1000    0.0848
##      6        1.1944             nan     0.1000    0.0630
##      7        1.1538             nan     0.1000    0.0647
##      8        1.1124             nan     0.1000    0.0611
##      9        1.0731             nan     0.1000    0.0458
##     10        1.0437             nan     0.1000    0.0493
##     20        0.8200             nan     0.1000    0.0315
##     40        0.5948             nan     0.1000    0.0109
##     60        0.4796             nan     0.1000    0.0083
##     80        0.3983             nan     0.1000    0.0053
##    100        0.3351             nan     0.1000    0.0042
##    120        0.2892             nan     0.1000    0.0032
##    140        0.2501             nan     0.1000    0.0021
##    150        0.2365             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1225
##      2        1.5272             nan     0.1000    0.0812
##      3        1.4725             nan     0.1000    0.0590
##      4        1.4316             nan     0.1000    0.0486
##      5        1.3992             nan     0.1000    0.0404
##      6        1.3720             nan     0.1000    0.0427
##      7        1.3451             nan     0.1000    0.0345
##      8        1.3224             nan     0.1000    0.0293
##      9        1.3040             nan     0.1000    0.0283
##     10        1.2862             nan     0.1000    0.0281
##     20        1.1479             nan     0.1000    0.0140
##     40        0.9980             nan     0.1000    0.0082
##     60        0.9011             nan     0.1000    0.0055
##     80        0.8304             nan     0.1000    0.0032
##    100        0.7712             nan     0.1000    0.0046
##    120        0.7241             nan     0.1000    0.0028
##    140        0.6826             nan     0.1000    0.0024
##    150        0.6652             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1660
##      2        1.5001             nan     0.1000    0.1201
##      3        1.4220             nan     0.1000    0.0874
##      4        1.3655             nan     0.1000    0.0796
##      5        1.3157             nan     0.1000    0.0629
##      6        1.2751             nan     0.1000    0.0560
##      7        1.2391             nan     0.1000    0.0504
##      8        1.2068             nan     0.1000    0.0486
##      9        1.1754             nan     0.1000    0.0391
##     10        1.1507             nan     0.1000    0.0454
##     20        0.9612             nan     0.1000    0.0202
##     40        0.7634             nan     0.1000    0.0097
##     60        0.6291             nan     0.1000    0.0074
##     80        0.5419             nan     0.1000    0.0041
##    100        0.4773             nan     0.1000    0.0044
##    120        0.4268             nan     0.1000    0.0036
##    140        0.3812             nan     0.1000    0.0024
##    150        0.3627             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2051
##      2        1.4766             nan     0.1000    0.1457
##      3        1.3834             nan     0.1000    0.1044
##      4        1.3135             nan     0.1000    0.1010
##      5        1.2501             nan     0.1000    0.0741
##      6        1.2019             nan     0.1000    0.0755
##      7        1.1532             nan     0.1000    0.0545
##      8        1.1180             nan     0.1000    0.0550
##      9        1.0834             nan     0.1000    0.0586
##     10        1.0477             nan     0.1000    0.0506
##     20        0.8266             nan     0.1000    0.0236
##     40        0.6039             nan     0.1000    0.0100
##     60        0.4833             nan     0.1000    0.0072
##     80        0.3957             nan     0.1000    0.0037
##    100        0.3349             nan     0.1000    0.0045
##    120        0.2890             nan     0.1000    0.0027
##    140        0.2531             nan     0.1000    0.0016
##    150        0.2376             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1220
##      2        1.5269             nan     0.1000    0.0825
##      3        1.4725             nan     0.1000    0.0598
##      4        1.4315             nan     0.1000    0.0492
##      5        1.3988             nan     0.1000    0.0453
##      6        1.3691             nan     0.1000    0.0368
##      7        1.3453             nan     0.1000    0.0323
##      8        1.3242             nan     0.1000    0.0322
##      9        1.3040             nan     0.1000    0.0270
##     10        1.2866             nan     0.1000    0.0299
##     20        1.1483             nan     0.1000    0.0151
##     40        0.9960             nan     0.1000    0.0076
##     60        0.8999             nan     0.1000    0.0050
##     80        0.8284             nan     0.1000    0.0056
##    100        0.7680             nan     0.1000    0.0034
##    120        0.7200             nan     0.1000    0.0031
##    140        0.6811             nan     0.1000    0.0023
##    150        0.6633             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1685
##      2        1.4995             nan     0.1000    0.1169
##      3        1.4235             nan     0.1000    0.0934
##      4        1.3635             nan     0.1000    0.0690
##      5        1.3179             nan     0.1000    0.0705
##      6        1.2734             nan     0.1000    0.0547
##      7        1.2381             nan     0.1000    0.0469
##      8        1.2078             nan     0.1000    0.0511
##      9        1.1756             nan     0.1000    0.0412
##     10        1.1488             nan     0.1000    0.0373
##     20        0.9620             nan     0.1000    0.0193
##     40        0.7625             nan     0.1000    0.0158
##     60        0.6315             nan     0.1000    0.0104
##     80        0.5465             nan     0.1000    0.0049
##    100        0.4818             nan     0.1000    0.0029
##    120        0.4308             nan     0.1000    0.0029
##    140        0.3867             nan     0.1000    0.0013
##    150        0.3695             nan     0.1000    0.0028
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1997
##      2        1.4777             nan     0.1000    0.1467
##      3        1.3828             nan     0.1000    0.1165
##      4        1.3076             nan     0.1000    0.0860
##      5        1.2530             nan     0.1000    0.0755
##      6        1.2053             nan     0.1000    0.0671
##      7        1.1600             nan     0.1000    0.0705
##      8        1.1158             nan     0.1000    0.0587
##      9        1.0786             nan     0.1000    0.0573
##     10        1.0431             nan     0.1000    0.0519
##     20        0.8231             nan     0.1000    0.0219
##     40        0.5977             nan     0.1000    0.0137
##     60        0.4768             nan     0.1000    0.0050
##     80        0.3938             nan     0.1000    0.0025
##    100        0.3340             nan     0.1000    0.0040
##    120        0.2859             nan     0.1000    0.0036
##    140        0.2492             nan     0.1000    0.0019
##    150        0.2333             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1184
##      2        1.5268             nan     0.1000    0.0843
##      3        1.4723             nan     0.1000    0.0609
##      4        1.4315             nan     0.1000    0.0490
##      5        1.3994             nan     0.1000    0.0403
##      6        1.3725             nan     0.1000    0.0408
##      7        1.3460             nan     0.1000    0.0364
##      8        1.3231             nan     0.1000    0.0275
##      9        1.3052             nan     0.1000    0.0307
##     10        1.2858             nan     0.1000    0.0316
##     20        1.1486             nan     0.1000    0.0182
##     40        0.9958             nan     0.1000    0.0075
##     60        0.8992             nan     0.1000    0.0076
##     80        0.8258             nan     0.1000    0.0047
##    100        0.7660             nan     0.1000    0.0029
##    120        0.7193             nan     0.1000    0.0022
##    140        0.6793             nan     0.1000    0.0027
##    150        0.6613             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1663
##      2        1.4990             nan     0.1000    0.1157
##      3        1.4236             nan     0.1000    0.0872
##      4        1.3657             nan     0.1000    0.0760
##      5        1.3164             nan     0.1000    0.0610
##      6        1.2762             nan     0.1000    0.0573
##      7        1.2399             nan     0.1000    0.0502
##      8        1.2071             nan     0.1000    0.0492
##      9        1.1756             nan     0.1000    0.0439
##     10        1.1477             nan     0.1000    0.0435
##     20        0.9587             nan     0.1000    0.0195
##     40        0.7586             nan     0.1000    0.0100
##     60        0.6310             nan     0.1000    0.0069
##     80        0.5443             nan     0.1000    0.0060
##    100        0.4785             nan     0.1000    0.0071
##    120        0.4249             nan     0.1000    0.0027
##    140        0.3824             nan     0.1000    0.0024
##    150        0.3647             nan     0.1000    0.0030
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2052
##      2        1.4746             nan     0.1000    0.1403
##      3        1.3854             nan     0.1000    0.1193
##      4        1.3094             nan     0.1000    0.0947
##      5        1.2493             nan     0.1000    0.0836
##      6        1.1935             nan     0.1000    0.0675
##      7        1.1499             nan     0.1000    0.0578
##      8        1.1134             nan     0.1000    0.0584
##      9        1.0760             nan     0.1000    0.0566
##     10        1.0412             nan     0.1000    0.0472
##     20        0.8133             nan     0.1000    0.0249
##     40        0.5966             nan     0.1000    0.0098
##     60        0.4764             nan     0.1000    0.0051
##     80        0.3954             nan     0.1000    0.0050
##    100        0.3329             nan     0.1000    0.0031
##    120        0.2852             nan     0.1000    0.0016
##    140        0.2485             nan     0.1000    0.0017
##    150        0.2321             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2100
##      2        1.4743             nan     0.1000    0.1449
##      3        1.3814             nan     0.1000    0.1115
##      4        1.3096             nan     0.1000    0.0846
##      5        1.2547             nan     0.1000    0.0762
##      6        1.2041             nan     0.1000    0.0743
##      7        1.1565             nan     0.1000    0.0581
##      8        1.1196             nan     0.1000    0.0666
##      9        1.0772             nan     0.1000    0.0517
##     10        1.0442             nan     0.1000    0.0451
##     20        0.8249             nan     0.1000    0.0225
##     40        0.5980             nan     0.1000    0.0123
##     60        0.4731             nan     0.1000    0.0064
##     80        0.3925             nan     0.1000    0.0052
##    100        0.3327             nan     0.1000    0.0034
##    120        0.2852             nan     0.1000    0.0016
##    140        0.2494             nan     0.1000    0.0025
##    150        0.2323             nan     0.1000    0.0012
```

I then selected the same variables in the test data set that were used to train my
model:

```r
#predict with model
test_data <- read.csv("pml-testing.csv")

#keeping only the variables from the train data model
keepvariables <- names(clean_data)

#removing the classe variable
keepvariables <- keepvariables[-1]

#subsetting the test data
clean_test_data <- subset(test_data, select = keepvariables)
```

After cleaning the test data set, I used the model to predict the values of the
20 test cases, which were then written to 20 text files.

```r
answers <- predict(model, clean_test_data)

pml_write_files <- function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
```

## Results
Below is the model that was used in prediction, which yielded correct predictions
for all 20 test cases:

```r
print(model)
```

```
## Stochastic Gradient Boosting 
## 
## 19622 samples
##    31 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17660, 17661, 17660, 17659, 17659, 17659, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   1                   50      0.7051777  0.6246798  0.006496168
##   1                  100      0.7806525  0.7220122  0.008624982
##   1                  150      0.8177036  0.7691102  0.009425137
##   2                   50      0.8346748  0.7907485  0.006671580
##   2                  100      0.8878296  0.8581058  0.009586367
##   2                  150      0.9097948  0.8858576  0.010332121
##   3                   50      0.8806951  0.8490438  0.009051787
##   3                  100      0.9238094  0.9035783  0.009127431
##   3                  150      0.9466919  0.9325506  0.006709044
##   Kappa SD   
##   0.008421923
##   0.010967448
##   0.011902878
##   0.008434047
##   0.012131857
##   0.013082334
##   0.011486971
##   0.011556881
##   0.008496667
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```
As the model incorporates 10-fold cross-validation into the train function that
was used to build it, the out of sample (OOS) error is:
**OOS = 1 - accuracy = 1-0.95 = 0.05**
