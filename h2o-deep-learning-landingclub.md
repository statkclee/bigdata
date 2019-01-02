---
layout: page
title: xwMOOC 딥러닝
subtitle: $H_2 O$ vs 딥러닝 랜딩클럽 대출 데이터
---

> ## 학습 목표 {.objectives}
>
> * $H_2 O$ 기계학습과 딥러닝 랜딩클럽 대출 데이터 예측 모형을 비교한다.




### 1. 랜딩클럽 대출 데이터

랜딩클럽 대출 데이터는 163,987건에 대한 대출 이력을 15개 변수로 구성되어 있다. [Landing Club 대출 데이터](https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv)는 GitHub에서 바로 다운로드 가능하다. 원데이터는 [Lending Club Statistics](https://www.lendingclub.com/info/download-data.action) 사이트에서 다운로드 가능하다.

~~~ {.r}
> dim(data)
[1] 163987     15
> names(data)
 [1] "loan_amnt"             "term"                  "int_rate"             
 [4] "emp_length"            "home_ownership"        "annual_inc"           
 [7] "purpose"               "addr_state"            "dti"                  
[10] "delinq_2yrs"           "revol_util"            "total_acc"            
[13] "bad_loan"              "longest_credit_length" "verification_status"  
~~~

### 2. 환경설정 및 랜딩클럽 데이터 모형적합 준비 [^h2o-landingclub]

[^h2o-landingclub]: [H2O Machine Learning Tutorial - Grid Search and Model Selection](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/grid-search-model-selection.R)

~~~ {.r}
##=========================================================================
## 01. H2O 설치: http://learn.h2o.ai/content/tutorials/ensembles-stacking/
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
#install.packages("h2o", repos=(c("http://s3.amazonaws.com/h2o-release/h2o/master/1497/R", getOption("repos"))))

library(devtools)
install_github("h2oai/h2o-3/h2o-r/ensemble/h2oEnsemble-package")
#install.packages("https://h2o-release.s3.amazonaws.com/h2o-ensemble/R/h2oEnsemble_0.1.8.tar.gz", repos = NULL)

#-------------------------------------------------------------------------
# 01.1. H2O 클러스터 환경설정
#-------------------------------------------------------------------------

library(h2o)
library(h2oEnsemble)  # This will load the `h2o` R package as well
h2o.init(nthreads = -1, ip = 'localhost', port = 54321, max_mem_size = '8g')  # Start an H2O cluster with nthreads = num cores on your machine
h2o.removeAll() # Clean slate - just in case the cluster was already running
#h2o.shutdown()

##=========================================================================
## 02. H2O 데이터 가져오기
##=========================================================================
loan_csv <- "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"
data <- h2o.importFile(loan_csv)  # 163,987 rows x 15 columns
dim(data)

data$bad_loan <- as.factor(data$bad_loan)
h2o.levels(data$bad_loan)

splits <- h2o.splitFrame(data = data, 
                         ratios = c(0.7, 0.15),
                         seed = 7) 
train <- splits[[1]]
valid <- splits[[2]]
test <- splits[[3]]

nrow(train)  # 114908
nrow(valid) # 24498
nrow(test)  # 24581

y <- "bad_loan"
x <- setdiff(names(data), c(y, "int_rate"))  # 이자율이 종속변수와 상관되어 제거
print(x)
~~~

### 3. GBM 초모수 세부조정 탐색

~~~ {.r}
##=========================================================================
## 03. 예측모형 개발 GBM
##=========================================================================

#-------------------------------------------------------------------------
# 3.1. GBM 데카르트 격자 탐색
#-------------------------------------------------------------------------
gbm_params1 <- list(learn_rate = c(0.01, 0.1),
                    max_depth = c(3, 5, 9),
                    sample_rate = c(0.8, 1.0),
                    col_sample_rate = c(0.2, 0.5, 1.0))

# Train and validate a grid of GBMs
gbm_grid1 <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid1",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params1)

# Get the grid results, sorted by AUC
gbm_gridperf1 <- h2o.getGrid(grid_id = "gbm_grid1", 
                             sort_by = "auc", 
                             decreasing = TRUE)
print(gbm_gridperf1)


#-------------------------------------------------------------------------
# 3.2. GBM 임의 격자 탐색
#-------------------------------------------------------------------------

# GBM hyperparamters
gbm_params2 <- list(learn_rate = seq(0.01, 0.1, 0.01),
                    max_depth = seq(2, 10, 1),
                    sample_rate = seq(0.5, 1.0, 0.1),
                    col_sample_rate = seq(0.1, 1.0, 0.1))
search_criteria2 <- list(strategy = "RandomDiscrete", 
                         max_models = 36)

# Train and validate a grid of GBMs
gbm_grid2 <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid2",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params2,
                      search_criteria = search_criteria2)

gbm_gridperf2 <- h2o.getGrid(grid_id = "gbm_grid2", 
                             sort_by = "auc", 
                             decreasing = TRUE)
print(gbm_gridperf2)

#-------------------------------------------------------------------------
# 3.3. 학습율 조정 및 속도 60초내 조정
#-------------------------------------------------------------------------
gbm_params <- list(learn_rate = seq(0.1, 0.3, 0.01),  #updated
                   max_depth = seq(2, 10, 1),
                   sample_rate = seq(0.9, 1.0, 0.05),  #updated
                   col_sample_rate = seq(0.1, 1.0, 0.1))
search_criteria <- list(strategy = "RandomDiscrete", 
                        max_runtime_secs = 60)  #updated


gbm_grid <- h2o.grid("gbm", x = x, y = y,
                     grid_id = "gbm_grid2",
                     training_frame = train,
                     validation_frame = valid,
                     ntrees = 100,
                     seed = 1,
                     hyper_params = gbm_params,
                     search_criteria = search_criteria2)

gbm_gridperf <- h2o.getGrid(grid_id = "gbm_grid2", 
                            sort_by = "auc", 
                            decreasing = TRUE)
print(gbm_gridperf)
~~~

~~~ {.output}
> print(gbm_gridperf)
H2O Grid Details
================

Grid ID: gbm_grid2 
Used hyper parameters: 
  -  sample_rate 
  -  max_depth 
  -  learn_rate 
  -  col_sample_rate 
Number of models: 72 
Number of failed models: 0 

Hyper-Parameter Search Summary: ordered by decreasing auc
  sample_rate max_depth learn_rate col_sample_rate          model_ids               auc
1        0.95         4       0.15             0.8 gbm_grid2_model_52 0.685815702294939
2         0.9         5       0.09             0.3 gbm_grid2_model_32 0.684989247370194
3        0.95         3       0.16             0.4 gbm_grid2_model_40 0.684655168053999
4         0.9         6       0.07             0.5 gbm_grid2_model_23 0.684644609819277
5        0.95         4       0.16             0.9 gbm_grid2_model_61  0.68434074338269

---
   sample_rate max_depth learn_rate col_sample_rate          model_ids               auc
67        0.95         7       0.27             0.9 gbm_grid2_model_66 0.661731541447547
68           1         9       0.25             0.3 gbm_grid2_model_62 0.654547440520621
69           1         8       0.29             0.3 gbm_grid2_model_42 0.654499751942164
70           1        10       0.21             0.8 gbm_grid2_model_47 0.652858895249867
71        0.95        10       0.19             0.6 gbm_grid2_model_41 0.650640226199091
72        0.95        10       0.22             0.6 gbm_grid2_model_65 0.643525785349522
~~~

#### 3.1. GBM 최적모형 개발

~~~ {.r}
#-------------------------------------------------------------------------
# 3.4. 최적 모형 선정
#-------------------------------------------------------------------------

# 타당도 AUC를 갖고 최적 모형 선정
best_gbm_model_id <- gbm_gridperf@model_ids[[1]]
best_gbm <- h2o.getModel(best_gbm_model_id)

# 모형 성능
best_gbm_perf <- h2o.performance(model = best_gbm, 
                                 newdata = test)
h2o.auc(best_gbm_perf)  # 0.683855910541
~~~

~~~ {.output}
> h2o.auc(best_gbm_perf)  # 0.683855910541
[1] 0.6839909
~~~

### 4. 딥러닝 초모수 세부조정 탐색

~~~ {.r}
##=========================================================================
## 04. 예측모형 개발 딥러닝
##=========================================================================

# 딥러닝 초모수
activation_opt <- c("Rectifier", "RectifierWithDropout", "Maxout", "MaxoutWithDropout")
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
hyper_params <- list(activation = activation_opt,
                     l1 = l1_opt,
                     l2 = l2_opt)
search_criteria <- list(strategy = "RandomDiscrete", 
                        max_runtime_secs = 120)


dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    grid_id = "dl_grid",
                    training_frame = train,
                    validation_frame = valid,
                    seed = 1,
                    hidden = c(10,10),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)

dl_gridperf <- h2o.getGrid(grid_id = "dl_grid", 
                           sort_by = "auc", 
                           decreasing = TRUE)
print(dl_gridperf)
~~~

~~~ {.output}
> print(dl_gridperf)
H2O Grid Details
================

Grid ID: dl_grid 
Used hyper parameters: 
  -  l2 
  -  l1 
  -  activation 
Number of models: 44 
Number of failed models: 0 

Hyper-Parameter Search Summary: ordered by decreasing auc
     l2    l1 activation        model_ids               auc
1 0.001 1e-05     Maxout dl_grid_model_32 0.680359814489279
2 0.001 1e-05  Rectifier dl_grid_model_22 0.677697991945115
3     0 0.001  Rectifier dl_grid_model_13 0.676589653115321
4     0     0  Rectifier dl_grid_model_30 0.674574685179257
5 0.001 1e-04  Rectifier dl_grid_model_11 0.674083203214399

---
      l2    l1           activation        model_ids               auc
39   0.1 1e-05    MaxoutWithDropout  dl_grid_model_1               0.5
40 1e-05   0.1 RectifierWithDropout dl_grid_model_15               0.5
41   0.1  0.01            Rectifier dl_grid_model_18 0.499928486439431
42   0.1   0.1               Maxout  dl_grid_model_4 0.499661937901428
43   0.1 0.001               Maxout dl_grid_model_36 0.498875658328552
44 1e-04   0.1               Maxout dl_grid_model_34 0.497445227143926
~~~

~~~ {.r}
#-------------------------------------------------------------------------
# 4.1. 최적 모형 선정
#-------------------------------------------------------------------------


# 타당도 AUC를 갖고 최적 모형 선정
best_dl_model_id <- dl_gridperf@model_ids[[1]]
best_dl <- h2o.getModel(best_dl_model_id)

# 모형 성능평가
best_dl_perf <- h2o.performance(model = best_dl, 
                                newdata = test)
h2o.auc(best_dl_perf)
~~~


~~~ {.output}
> h2o.auc(best_dl_perf)  # .683855910541
[1] 0.6835463
~~~

결론은 GBM과 딥러닝 별차이가 없다.