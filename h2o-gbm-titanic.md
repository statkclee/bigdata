---
layout: page
title: xwMOOC 딥러닝
subtitle: $H_2 O$ GBM 모형 세부조정
---

> ## 학습 목표 {.objectives}
>
> * $H_2 O$ 활용 최적 예측모형을 개발한다.
> * 타이타닉 생존 데이터로 동일한 방법론을 갈음한다.
> * GBM 기본 모형을 바탕으로 모수 세부조정을 통해 성능을 높인다.

이제부터 타이타닉 생존 정확하게 예측할 수 있는 정말 정확도 높은 예측모형 개발을 위한 눈물겨운 여정을 떠나본다.
GBM을 가지고 AUC 0.94가 나오는데 다양한 초모수 세부조정을 통해 0.97까지 높일 수 있다. 물론 상위 10개 앙상블을 사용한다면 0.975까지도 가능하다.

<img src="fig/h2o-hyper-parameter-tuning.png" alt="H2O 초모수 세부조정 최적화" width="70%">


### 1. 타이타닉 생존 데이터 GBM 기본 모형구축 [^h2o-gbm-tuning]

[^h2o-gbm-tuning]: [H2O GBM Tuning Tutorial for R](http://blog.h2o.ai/2016/06/h2o-gbm-tuning-tutorial-for-r/)

1. $H_2 O$ 팩키지를 설치하고, $H_2 O$ 클러스터를 생성한다.
1. [타이타닉 생존 데이터](http://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv)를 다운로드 한다.
1. `h2o.splitFrame` 명령어를 통해 훈련, 타당도검증, 검증 데이터로 구분하고 GBM 모형을 적합시킨다.

~~~ {.r}
##=========================================================================
## 01. H2O 설치: http://blog.h2o.ai/2016/06/h2o-gbm-tuning-tutorial-for-r/
##=========================================================================
# 1. 기존 H2O 제거
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# 2. H2O 의존성 설치
pkgs <- c("methods","statmod","stats","graphics","RCurl","jsonlite","tools","utils")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}

# 3. H2O 설치
install.packages("h2o", repos=(c("http://s3.amazonaws.com/h2o-release/h2o/master/1497/R", getOption("repos"))))

#-------------------------------------------------------------------------
# 01.1. H2O 클러스터 환경설정
#-------------------------------------------------------------------------

library(h2o)
h2o.init(nthreads=-1)

##=========================================================================
## 02. H2O 데이터 가져오기
##=========================================================================

df <- h2o.importFile(path = "http://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")
summary(df, exact_quantiles=TRUE)

##=========================================================================
## 03. H2O 데이터 정제 과정
##=========================================================================

# 데이터 정제 및 변환과정 생략

##=========================================================================
## 04. GBM 모형
##=========================================================================
# 1. 종속변수 선정
response <- "survived"
# 2. 종속변수가 숫자형이라 요인(Factor)으로 자료형 변환
df[[response]] <- as.factor(df[[response]])           

# 3. 종속변수를 제외한 모든 변수를 설명변수
predictors <- setdiff(names(df), c(response, "name")) 

# 4. 훈련, 타당성검증, 검증 데이터로 분리
splits <- h2o.splitFrame(
  data = df, 
  ratios = c(0.6,0.2),   ## 60% 훈련, 20% 타당도검증, 나머지 20% 자동 검증데이터 생성
  destination_frames = c("train.hex", "valid.hex", "test.hex"), seed = 1234
)
train <- splits[[1]]
valid <- splits[[2]]
test  <- splits[[3]]

#--------------------------------------------------------------------------
# 4.1. GBM 초기 모형
#--------------------------------------------------------------------------
# 1. 기본 모형 생성
gbm <- h2o.gbm(x = predictors, y = response, training_frame = train)
gbm

# 타당성검증 데이터 AUC 성능
h2o.auc(h2o.performance(gbm, newdata = valid)) 

# 2. 타당성 검증 데이터 활용 GBM 모형 개발 
gbm <- h2o.gbm(x = predictors, y = response, training_frame = h2o.rbind(train, valid), nfolds = 4, seed = 0xDECAF)

gbm@model$cross_validation_metrics_summary
h2o.auc(h2o.performance(gbm, xval = TRUE))
~~~

~~~ {.output}
> gbm
[1] 0.9431953
> gbm@model$cross_validation_metrics_summary
Cross-Validation Metrics Summary: 
                           mean           sd  cv_1_valid cv_2_valid  cv_3_valid cv_4_valid
F0point5              0.9127705  0.010779412  0.92045456  0.9183673   0.8867521  0.9255079
F1                    0.8876407 0.0054373597   0.8756757  0.8959276   0.8924731  0.8864865
F2                   0.86462516  0.016960286  0.83505154  0.8745583   0.8982684  0.8506224
accuracy              0.9174827 0.0030382033   0.9151291  0.9118774   0.9230769  0.9198473
auc                   0.9432912  0.007747688   0.9298538 0.93615246  0.95788044 0.94927806
err                   0.0825173 0.0030382033 0.084870845 0.08812261  0.07692308 0.08015267
err_count                 21.75   0.91855866          23         23          20         21
lift_top_group        2.6130292   0.14742896        2.71   2.269565    2.826087  2.6464646
logloss              0.25946963  0.016240774  0.25494286 0.29227072  0.22781134 0.26285362
max_per_class_error  0.14966843  0.024777662        0.19 0.13913043 0.097826086 0.17171717
mcc                  0.82561684 0.0041895183  0.81807023  0.8217822  0.83272034  0.8298947
mse                 0.071905546 0.0039584376  0.07185569 0.07921115  0.06349978 0.07305556
precision            0.93084264  0.020315822   0.9529412  0.9339623  0.88297874 0.95348835
r2                    0.6953803  0.011497839   0.6913945  0.6786216   0.7222706  0.6892343
recall                0.8503316  0.024777662        0.81  0.8608696  0.90217394 0.82828283
specificity           0.9596617  0.012382206   0.9766082  0.9520548   0.9345238  0.9754601
~~~

### 2. GBM 모수로 설정한 값이 운좋게 최적일 수 있다.

`h2o.gbm` 모형에 설정한 값이 운좋게도 가장 최적일 수도 있다. 하지만, 그런 경우는 거의 없다.


~~~ {.r}
#--------------------------------------------------------------------------
# 4.2. GBM 정말 운좋은 모형 개발
#--------------------------------------------------------------------------
gbm <- h2o.gbm(
  ## 표준 모형 모수 설정
  x = predictors, 
  y = response, 
  training_frame = train, 
  validation_frame = valid,
  
  ntrees = 10000,                                                            
  
  learn_rate=0.01,                                                         

  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 

  sample_rate = 0.8,                                                       
  col_sample_rate = 0.8,                                                   

  seed = 1234,                                                             
  score_tree_interval = 10                                                 
)

h2o.auc(h2o.performance(gbm, valid = TRUE))
~~~

~~~ {.output}
> h2o.auc(h2o.performance(gbm, valid = TRUE))
[1] 0.939335
~~~

### 3. 초모수(Hyper-parameter) 설정을 통한 GBM 최적 모형 개발

최적의 GBM 모형 구축을 위해 초모수를 최적화하는 기계적인 방법은 존재하지 않으며, 
경험에 비추어 다음 모수가 최적 GBM 구축에 도움이 되는 것으로 알려져 있다.

1. `ntrees`: 타당도 검증 오차가 증가할 때까지 가능함 많은 나무모형을 생성시킨다.
1. `learn_rate`: 가능하면 낮은 학습율을 지정한다. 하지만 댓가로 더 많은 나무모형이 필요하다. `learn_rate=0.02`와 `learn_rate_annealing=0.995` 모수로 설정한다.
1. `max_depth`: 나무 깊이는 데이터에 따라 최적 깊이가 달라진다. 더 깊은 나무를 생성시키려면 더 많은 시간이 소요된다. 특히 10보다 큰 경우 깊은 나무모형으로 알려져 있다.
1. `sample_rate`, `col_sample_rate` : 행과 열을 표집추출하는 것으로 보통 0.7 -- 0.8 이 무난하다.
1. `sample_rate_per_class`: 심각한 불균형 데이터(예를 들어, 연체고객과 정상고객, 이상거래와 정상거래 등)의 경우 층화추출법을 통해 모형 정확도를 높일 수 있다.
1. 기타 나머지 모수는 상대적으로 적은 기여도를 보이는데, 필요한 경우 임의 초모수 검색법을 도모할 수 있다.

`max_depth`를 먼저 상정할 수 있고, 데카르트 좌표계(Cartesian Grid) 검색으로 이를 구현한다.


~~~ {.r}
#--------------------------------------------------------------------------
# 4.3. GBM 모수 미세조정
#--------------------------------------------------------------------------
hyper_params = list( max_depth = seq(1,29,2) )
#hyper_params = list( max_depth = c(4,6,8,12,16,20) ) ## 빅데이터의 경우 사용

grid <- h2o.grid(

  hyper_params = hyper_params,
  search_criteria = list(strategy = "Cartesian"),

  algorithm="gbm",
  grid_id="depth_grid",
  
  x = predictors, 
  y = response, 
  training_frame = train, 
  validation_frame = valid,
  

  ntrees = 10000,                                                            
  learn_rate = 0.05,                                                         
  learn_rate_annealing = 0.99,                                               
  
  sample_rate = 0.8,                                                       
  col_sample_rate = 0.8, 
  
  seed = 1234,                                                             
  
  stopping_rounds = 5,
  stopping_tolerance = 1e-4,
  stopping_metric = "AUC", 
  
  score_tree_interval = 10                                                
)

grid                                                                       

sortedGrid <- h2o.getGrid("depth_grid", sort_by="auc", decreasing = TRUE)    
sortedGrid


topDepths = sortedGrid@summary_table$max_depth[1:5]                       
minDepth = min(as.numeric(topDepths))
maxDepth = max(as.numeric(topDepths))
~~~

~~~ {.output}
> sortedGrid
H2O Grid Details
================

Grid ID: depth_grid 
Used hyper parameters: 
  -  max_depth 
Number of models: 15 
Number of failed models: 0 

Hyper-Parameter Search Summary: ordered by decreasing auc
   max_depth           model_ids               auc
1         27 depth_grid_model_13  0.95657931811778
2         25 depth_grid_model_12 0.956353902507749
3         29 depth_grid_model_14 0.956241194702733
4         21 depth_grid_model_10 0.954663285432516
5         19  depth_grid_model_9 0.954494223724993
6         13  depth_grid_model_6 0.954381515919978
7         23 depth_grid_model_11 0.954043392504931
8         11  depth_grid_model_5 0.952183713722175
9         15  depth_grid_model_7 0.951789236404621
10        17  depth_grid_model_8 0.951507466892082
11         9  depth_grid_model_4 0.950436742744435
12         7  depth_grid_model_3 0.946942800788955
13         5  depth_grid_model_2 0.939306846999155
14         3  depth_grid_model_1 0.932713440405748
15         1  depth_grid_model_0  0.92902225979149
~~~

### 4. GBM 초모수 격자탐색

`hyper_params` 리스트에 격자 탐색할 모수를 설정하고, `search_criteria`에 탐색기준을 적시하고 나서
`h2o.grid`를 통해 최적 GBM 초모수를 탐색한다.

~~~ {.r}
#--------------------------------------------------------------------------
# 4.4. GBM 초모수 격자 탐색
#--------------------------------------------------------------------------

hyper_params = list( 

  max_depth = seq(minDepth,maxDepth,1),                                      
  
  sample_rate = seq(0.2,1,0.01),                                             
  
  col_sample_rate = seq(0.2,1,0.01),                                         
  
  col_sample_rate_per_tree = seq(0.2,1,0.01),                                
  
  col_sample_rate_change_per_level = seq(0.9,1.1,0.01),                      
  
  min_rows = 2^seq(0,log2(nrow(train))-1,1),                                 
  
  nbins = 2^seq(4,10,1),                                                     
  
  nbins_cats = 2^seq(4,12,1),                                                
  
  min_split_improvement = c(0,1e-8,1e-6,1e-4),                               
  
  histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin")       
)

search_criteria = list(
  strategy = "RandomDiscrete",      
  
  max_runtime_secs = 3600,         
  
  max_models = 100,                  
  
  seed = 1234,                        
  
  stopping_rounds = 5,                
  stopping_metric = "AUC",
  stopping_tolerance = 1e-3
)

grid <- h2o.grid(

  hyper_params = hyper_params,
  search_criteria = search_criteria,
  
  algorithm = "gbm",
  
  grid_id = "final_grid", 
  
  x = predictors, 
  y = response, 
  training_frame = train, 
  validation_frame = valid,

  ntrees = 10000,                                                            
  
  learn_rate = 0.05,                                                         
  
  learn_rate_annealing = 0.99,                                               
  
  max_runtime_secs = 3600,                                                 
  
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 
  
  score_tree_interval = 10,                                                
  
  seed = 1234                                                             
)

## AUC 기준 격자모형 정렬
sortedGrid <- h2o.getGrid("final_grid", sort_by = "auc", decreasing = TRUE)    
sortedGrid

for (i in 1:5) {
  gbm <- h2o.getModel(sortedGrid@model_ids[[i]])
  print(h2o.auc(h2o.performance(gbm, valid = TRUE)))
}
~~~

~~~ {.output}
> sortedGrid
H2O Grid Details
================

Grid ID: final_grid 
Used hyper parameters: 
  -  histogram_type 
  -  sample_rate 
  -  nbins_cats 
  -  nbins 
  -  min_rows 
  -  col_sample_rate_change_per_level 
  -  min_split_improvement 
  -  max_depth 
  -  col_sample_rate 
  -  col_sample_rate_per_tree 
Number of models: 199 
Number of failed models: 0 

Hyper-Parameter Search Summary: ordered by decreasing auc
  histogram_type sample_rate nbins_cats nbins min_rows  ... max_depth
1     RoundRobin         0.9        256  1024        8  ...        28
2     RoundRobin         0.9        256  1024        8  ...        28
3     RoundRobin        0.39        256   512        4  ...        21
4     RoundRobin        0.39        256   512        4  ...        21
5     RoundRobin        0.31        256   128        2  ...        20
  col_sample_rate ...            model_ids               auc
1            0.65 ...  final_grid_model_59 0.970498732037194
2            0.65 ... final_grid_model_159 0.970498732037194
3            0.43 ... final_grid_model_142 0.969822485207101
4            0.43 ...  final_grid_model_42 0.969822485207101
5            0.37 ... final_grid_model_172 0.969202592279515

---
     histogram_type sample_rate nbins_cats nbins min_rows ... min_split_improvement
194 UniformAdaptive        0.96       2048   512      256 ...                 1e-08
195 UniformAdaptive        0.96       2048   512      256 ...                 1e-08
196 UniformAdaptive        0.82         64    16      256 ...                 1e-04
197 UniformAdaptive        0.82         64    16      256 ...                 1e-04
198 QuantilesGlobal        0.64        512    32      256 ...                 1e-08
199 QuantilesGlobal        0.64        512    32      256 ...                 1e-08
    max_depth col_sample_rate ...            model_ids               auc
194        28            0.56 ...  final_grid_model_58 0.794449140602987
195        28            0.56 ... final_grid_model_158 0.794449140602987
196        21             0.5 ... final_grid_model_189 0.791180614257537
197        21             0.5 ...  final_grid_model_89 0.791180614257537
198        19             0.9 ...  final_grid_model_64 0.741617357001972
199        19             0.9 ... final_grid_model_164 0.741617357001972
~~~

### 5. 최종모형 정리

AUC 기준 가장 좋은 모형을 하나 선정하고, 이를 기준으로 검증데이터 혹은 예측이 필요한 데이터에 예측확률을 붙여 저장한다.

~~~ {.r}
##=========================================================================
## 05. 최종 모형 정리
##=========================================================================

gbm <- h2o.getModel(sortedGrid@model_ids[[1]])
print(h2o.auc(h2o.performance(gbm, newdata = test)))

gbm@parameters

model <- do.call(h2o.gbm,
                 ## update parameters in place
                 {
                   p <- gbm@parameters
                   p$model_id = NULL          ## do not overwrite the original grid model
                   p$training_frame = df      ## use the full dataset
                   p$validation_frame = NULL  ## no validation frame
                   p$nfolds = 5               ## cross-validation
                   p
                 }
)
model@model$cross_validation_metrics_summary

for (i in 1:5) {
  gbm <- h2o.getModel(sortedGrid@model_ids[[i]])
  cvgbm <- do.call(h2o.gbm,
                   ## update parameters in place
                   {
                     p <- gbm@parameters
                     p$model_id = NULL          ## do not overwrite the original grid model
                     p$training_frame = df      ## use the full dataset
                     p$validation_frame = NULL  ## no validation frame
                     p$nfolds = 5               ## cross-validation
                     p
                   }
  )
  print(gbm@model_id)
  print(cvgbm@model$cross_validation_metrics_summary[5,]) ## Pick out the "AUC" row
}

gbm <- h2o.getModel(sortedGrid@model_ids[[1]])
preds <- h2o.predict(gbm, test)
head(preds)
gbm@model$validation_metrics@metrics$max_criteria_and_metric_scores


h2o.saveModel(gbm, "~/30-neural-network/bestModel.csv", force=TRUE)
h2o.exportFile(preds, "~/30-neural-network/bestPreds.csv", force=TRUE)
~~~

~~~ {.output}
> print(h2o.auc(h2o.performance(gbm, newdata = test)))
[1] 0.9743624
> 
> gbm@parameters
$model_id
[1] "final_grid_model_59"

$training_frame
[1] "train.hex"

$validation_frame
[1] "valid.hex"

$score_tree_interval
[1] 10

$ntrees
[1] 10000

$max_depth
[1] 28

$min_rows
[1] 8

$nbins
[1] 1024

$nbins_cats
[1] 256

$stopping_rounds
[1] 5

$stopping_metric
[1] "AUC"

$stopping_tolerance
[1] 1e-04

$max_runtime_secs
[1] 3461.373

$seed
[1] 1234

$learn_rate
[1] 0.05

$learn_rate_annealing
[1] 0.99

$distribution
[1] "bernoulli"

$sample_rate
[1] 0.9

$col_sample_rate
[1] 0.65

$col_sample_rate_change_per_level
[1] 1.02

$col_sample_rate_per_tree
[1] 0.67

$histogram_type
[1] "RoundRobin"

$x
 [1] "pclass"    "sex"       "age"       "sibsp"     "parch"     "ticket"    "fare"      "cabin"     "embarked" 
[10] "boat"      "body"      "home.dest"

$y
[1] "survived"

> model@model$cross_validation_metrics_summary
Cross-Validation Metrics Summary: 
                           mean           sd  cv_1_valid cv_2_valid  cv_3_valid  cv_4_valid  cv_5_valid
F0point5             0.93560344  0.014515156   0.9448819  0.9404762  0.89641434   0.9567198   0.9395248
F1                    0.9102194  0.008333748   0.9099526  0.8926554   0.9045226   0.9281768   0.9157895
F2                    0.8868533  0.015512505   0.8775137  0.8494624   0.9127789  0.90128756   0.8932238
accuracy             0.93442553 0.0058180066  0.92883897  0.9298893   0.9263566   0.9488189  0.93822396
auc                   0.9695648 0.0060331114   0.9673963  0.9566369   0.9658801  0.98000664    0.977904
err                  0.06557447 0.0058180066  0.07116105  0.0701107  0.07364341 0.051181104  0.06177606
err_count                  17.2    1.6970563          19         19          19          13          16
lift_top_group        2.6258688  0.099894695   2.3839285  2.8229167    2.632653   2.6736841   2.6161616
logloss              0.19936788  0.015208231  0.21330717 0.22592694   0.2054847  0.16403295  0.18808767
max_per_class_error  0.12771495  0.022303823  0.14285715 0.17708333  0.08163265  0.11578947 0.121212125
mcc                   0.8617862  0.011978933   0.8559271   0.847847   0.8448622   0.8912296   0.8690649
mse                 0.055098847 0.0046449783 0.059775334 0.06225182 0.058212284  0.04435746  0.05089733
precision             0.9537766  0.022758426    0.969697 0.97530866   0.8910891   0.9767442  0.95604396
r2                     0.766055   0.02020471  0.75453204  0.7278669   0.7528799   0.8105418   0.7844543
recall               0.87228507  0.022303823  0.85714287  0.8229167   0.9183673   0.8842105   0.8787879
specificity           0.9725776  0.015016872   0.9806452  0.9885714     0.93125   0.9874214       0.975

> gbm@model$validation_metrics@metrics$max_criteria_and_metric_scores
Maximum Metrics: Maximum metrics at their respective thresholds
                      metric threshold    value idx
1                     max f1  0.421230 0.935961  97
2                     max f2  0.262753 0.928030 107
3               max f0point5  0.729648 0.962801  87
4               max accuracy  0.478497 0.952555  95
5              max precision  0.988875 1.000000   0
6                 max recall  0.013276 1.000000 252
7            max specificity  0.988875 1.000000   0
8           max absolute_MCC  0.478497 0.900226  95
9 max min_per_class_accuracy  0.262753 0.933333 107
~~~

초모수 미세조정을 통해 최적화된 GBM 모형 하나보다 경우에 따라서는 앙상블 기법을 사용한 방법이 더 좋은 성능을 보여주기도 한다.

~~~ {.r}
#--------------------------------------------------------------------------
# 5.1. 앙상블 기법
#--------------------------------------------------------------------------

prob = NULL
k=10
for (i in 1:k) {
  gbm <- h2o.getModel(sortedGrid@model_ids[[i]])
  if (is.null(prob)) prob = h2o.predict(gbm, test)$p1
  else prob = prob + h2o.predict(gbm, test)$p1
}
prob <- prob/k
head(prob)

probInR  <- as.vector(prob)
labelInR <- as.vector(as.numeric(test[[response]]))
if (! ("cvAUC" %in% rownames(installed.packages()))) { install.packages("cvAUC") }
library(cvAUC)
cvAUC::AUC(probInR, labelInR)
~~~

~~~ {.output}
> cvAUC::AUC(probInR, labelInR)
[1] 0.9748249
~~~