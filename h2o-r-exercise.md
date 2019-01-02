# xwMOOC 딥러닝




## 1. $H_2 O$ 연습문제 1부 {#h2o-exercise-part-1}

$H_2 O$ 학습기는 최신 자바를 설치를 요구하고 있다. 따라서 자바를 설치하고 나서 
`h2o` 팩키지를 설치한다. 연습문제 영어 원본은  
[Data Analytics with H20 in R Exercises -Part 1](http://www.r-exercises.com/2017/09/22/big-data-analytics-with-h20-in-r-exercises-part-1/),
해답은 [Data Analytics with H20 in R Exercises Part 1 Solution](http://www.r-exercises.com/2017/09/22/big-data-analytics-with-h20-in-r-exercises-part-1-solution/)을 참조한다.


## 1.1. 연습문제  {#h2o-exercise-part-1-ex01}

 H2O 다운로드 후 클러스트 초기화한다.


~~~{.r}
library(h2o)

h2o.cluster <- h2o.init()
~~~



~~~{.output}
 Connection successful!

R is connected to the H2O cluster: 
    H2O cluster uptime:         53 minutes 48 seconds 
    H2O cluster version:        3.10.5.3 
    H2O cluster version age:    2 months and 23 days  
    H2O cluster name:           H2O_started_from_R_KwangChun_jwk366 
    H2O cluster total nodes:    1 
    H2O cluster total memory:   1.58 GB 
    H2O cluster total cores:    4 
    H2O cluster allowed cores:  4 
    H2O cluster healthy:        TRUE 
    H2O Connection ip:          localhost 
    H2O Connection port:        54321 
    H2O Connection proxy:       NA 
    H2O Internal Security:      FALSE 
    R Version:                  R version 3.4.1 (2017-06-30) 

~~~

## 1.2. 연습문제 {#h2o-exercise-part-1-ex02}

clusterinfo를 통해 클러스터 정보를 확인한다.


~~~{.r}
h2o.clusterInfo()
~~~



~~~{.output}
R is connected to the H2O cluster: 
    H2O cluster uptime:         53 minutes 49 seconds 
    H2O cluster version:        3.10.5.3 
    H2O cluster version age:    2 months and 23 days  
    H2O cluster name:           H2O_started_from_R_KwangChun_jwk366 
    H2O cluster total nodes:    1 
    H2O cluster total memory:   1.58 GB 
    H2O cluster total cores:    4 
    H2O cluster allowed cores:  4 
    H2O cluster healthy:        TRUE 
    H2O Connection ip:          localhost 
    H2O Connection port:        54321 
    H2O Connection proxy:       NA 
    H2O Internal Security:      FALSE 
    R Version:                  R version 3.4.1 (2017-06-30) 

~~~

## 1.3. 연습문제  {#h2o-exercise-part-1-ex03}

H2O 작업을 `deom` 함수를 통해서 작업할 수 있다. H2O glm 모형을 살펴보자.


~~~{.r}
demo(h2o.glm)
~~~



~~~{.output}


	demo(h2o.glm)
	---- ~~~~~~~

> # This is a demo of H2O's GLM function
> # It imports a data set, parses it, and prints a summary
> # Then, it runs GLM with a binomial link function using 10-fold cross-validation
> # Note: This demo runs H2O on localhost:54321
> library(h2o)

> h2o.init()
 Connection successful!

R is connected to the H2O cluster: 
    H2O cluster uptime:         53 minutes 49 seconds 
    H2O cluster version:        3.10.5.3 
    H2O cluster version age:    2 months and 23 days  
    H2O cluster name:           H2O_started_from_R_KwangChun_jwk366 
    H2O cluster total nodes:    1 
    H2O cluster total memory:   1.58 GB 
    H2O cluster total cores:    4 
    H2O cluster allowed cores:  4 
    H2O cluster healthy:        TRUE 
    H2O Connection ip:          localhost 
    H2O Connection port:        54321 
    H2O Connection proxy:       NA 
    H2O Internal Security:      FALSE 
    R Version:                  R version 3.4.1 (2017-06-30) 


> prostate.hex = h2o.uploadFile(path = system.file("extdata", "prostate.csv", package="h2o"), destination_frame = "prostate.hex")

  |                                                                       
  |                                                                 |   0%
  |                                                                       
  |=================================================================| 100%

> summary(prostate.hex)
 ID               CAPSULE          AGE             RACE           
 Min.   :  1.00   Min.   :0.0000   Min.   :43.00   Min.   :0.000  
 1st Qu.: 95.75   1st Qu.:0.0000   1st Qu.:62.00   1st Qu.:1.000  
 Median :190.50   Median :0.0000   Median :67.00   Median :1.000  
 Mean   :190.50   Mean   :0.4026   Mean   :66.04   Mean   :1.087  
 3rd Qu.:285.25   3rd Qu.:1.0000   3rd Qu.:71.00   3rd Qu.:1.000  
 Max.   :380.00   Max.   :1.0000   Max.   :79.00   Max.   :2.000  
 DPROS           DCAPS           PSA               VOL            
 Min.   :1.000   Min.   :1.000   Min.   :  0.300   Min.   : 0.00  
 1st Qu.:1.000   1st Qu.:1.000   1st Qu.:  4.900   1st Qu.: 0.00  
 Median :2.000   Median :1.000   Median :  8.664   Median :14.20  
 Mean   :2.271   Mean   :1.108   Mean   : 15.409   Mean   :15.81  
 3rd Qu.:3.000   3rd Qu.:1.000   3rd Qu.: 17.063   3rd Qu.:26.40  
 Max.   :4.000   Max.   :2.000   Max.   :139.700   Max.   :97.60  
 GLEASON        
 Min.   :0.000  
 1st Qu.:6.000  
 Median :6.000  
 Mean   :6.384  
 3rd Qu.:7.000  
 Max.   :9.000  

> prostate.glm = h2o.glm(x = c("AGE","RACE","PSA","DCAPS"), y = "CAPSULE", training_frame = prostate.hex, family = "binomial", alpha = 0.5)

  |                                                                       
  |                                                                 |   0%
  |                                                                       
  |=================================================================| 100%

> print(prostate.glm)
Model Details:
==============

H2OBinomialModel: glm
Model ID:  GLM_model_R_1506128894010_7 
GLM Model: summary
    family  link                                regularization
1 binomial logit Elastic Net (alpha = 0.5, lambda = 3.247E-4 )
  number_of_predictors_total number_of_active_predictors
1                          4                           4
  number_of_iterations training_frame
1                    4   prostate.hex

Coefficients: glm coefficients
      names coefficients standardized_coefficients
1 Intercept    -1.114418                 -0.337704
2       AGE    -0.010977                 -0.071648
3      RACE    -0.623216                 -0.192433
4     DCAPS     1.314591                  0.408386
5       PSA     0.046892                  0.937727

H2OBinomialMetrics: glm
** Reported on training data. **

MSE:  0.2027036
RMSE:  0.4502261
LogLoss:  0.5914634
Mean Per-Class Error:  0.3826121
AUC:  0.717601
Gini:  0.435202
R^2:  0.1572256
Residual Deviance:  449.5122
AIC:  459.5122

Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
        0   1    Error      Rate
0      80 147 0.647577  =147/227
1      18 135 0.117647   =18/153
Totals 98 282 0.434211  =165/380

Maximum Metrics: Maximum metrics at their respective thresholds
                        metric threshold    value idx
1                       max f1  0.284048 0.620690 274
2                       max f2  0.207093 0.778230 360
3                 max f0point5  0.413268 0.636672 108
4                 max accuracy  0.413268 0.705263 108
5                max precision  0.998478 1.000000   0
6                   max recall  0.207093 1.000000 360
7              max specificity  0.998478 1.000000   0
8             max absolute_mcc  0.413268 0.369123 108
9   max min_per_class_accuracy  0.331806 0.647577 176
10 max mean_per_class_accuracy  0.373175 0.672123 126

Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`



> myLabels = c(prostate.glm@model$x, "Intercept")

> plot(prostate.glm@model$coefficients, xaxt = "n", xlab = "Coefficients", ylab = "Values")

~~~

<img src="fig/h2o-glm-demo-1.png" style="display: block; margin: auto;" />

~~~{.output}

> axis(1, at = 1:length(myLabels), labels = myLabels)

> abline(h = 0, col = 2, lty = 2)

> title("Coefficients from Logistic Regression\n of Prostate Cancer Data")

> barplot(prostate.glm@model$coefficients, main = "Coefficients from Logistic Regression\n of Prostate Cancer Data")

~~~

<img src="fig/h2o-glm-demo-2.png" style="display: block; margin: auto;" />

## 1.4. 연습문제 {#h2o-exercise-part-1-ex04}

H2O 깃헙에서 `load.csv` 파일을 다운로드 받아 이를 H2O로 가져온다.
파일 위치: [loan.csv](https://raw.githubusercontent.com/h2oai/h2o-tutorials/master/training/data/loan.csv), 약 16MB.


~~~{.r}
loan.hex <- h2o.importFile(path = normalizePath("data/loan.csv"))
~~~



~~~{.output}

  |                                                                       
  |                                                                 |   0%
  |                                                                       
  |=================================================================| 100%

~~~

## 1.5. 연습문제 {#h2o-exercise-part-1-ex05}

H2O 클러스터로 불러온 대출 데이터 자료형을 파악한다. 데이터프레임가 아니라는 사실을 유념한다. 대출데이터를 요약한다.


힌트: `h2o.summary()` 함수를 사용한다.


~~~{.r}
class(loan.hex)
~~~



~~~{.output}
[1] "H2OFrame"

~~~



~~~{.r}
h2o.summary(loan.hex)
~~~



~~~{.output}
 loan_amnt       term              int_rate        emp_length      
 Min.   :  500   36 months:129950  Min.   : 5.42   Min.   : 0.000  
 1st Qu.: 6986   60 months: 34037  1st Qu.:10.64   1st Qu.: 2.000  
 Median :11299                     Median :13.47   Median : 6.000  
 Mean   :13074                     Mean   :13.72   Mean   : 5.684  
 3rd Qu.:17992                     3rd Qu.:16.32   3rd Qu.:10.000  
 Max.   :35000                     Max.   :26.06   Max.   :10.000  
                                                   NA's   :5804    
 home_ownership  annual_inc        purpose                   addr_state
 MORTGAGE:79714  Min.   :   1896   debt_consolidation:93261  CA:28702  
 RENT    :70526  1st Qu.:  44735   credit_card       :30792  NY:14285  
 OWN     :13560  Median :  59015   other             :10492  TX:12128  
 OTHER   :  156  Mean   :  71916   home_improvement  : 9872  FL:11396  
 NONE    :   30  3rd Qu.:  80435   major_purchase    : 4686  NJ: 6457  
 ANY     :    1  Max.   :7141778   small_business    : 3841  IL: 6099  
                 NA's   :4                                             
 dti             delinq_2yrs       revol_util       total_acc       
 Min.   : 0.00   Min.   : 0.0000   Min.   :  0.00   Min.   :  1.00  
 1st Qu.:10.20   1st Qu.: 0.0000   1st Qu.: 35.57   1st Qu.: 16.00  
 Median :15.60   Median : 0.0000   Median : 55.76   Median : 23.00  
 Mean   :15.88   Mean   : 0.2274   Mean   : 54.08   Mean   : 24.58  
 3rd Qu.:21.23   3rd Qu.: 0.0000   3rd Qu.: 74.14   3rd Qu.: 31.00  
 Max.   :39.99   Max.   :29.0000   Max.   :150.70   Max.   :118.00  
                 NA's   :29        NA's   :193      NA's   :29      
 bad_loan        longest_credit_length verification_status 
 Min.   :0.000   Min.   : 0.00         verified    :104832 
 1st Qu.:0.000   1st Qu.:10.00         not verified: 59155 
 Median :0.000   Median :14.00                             
 Mean   :0.183   Mean   :14.85                             
 3rd Qu.:0.000   3rd Qu.:18.00                             
 Max.   :1.000   Max.   :65.00                             
                 NA's   :29                                

~~~


## 1.6. 연습문제 {#h2o-exercise-part-1-ex06}

R 환경에 데이터프레임을 H2O 클러스터로 데이터 이전해야 하는 경우가 있다. `as.h2o` 함수를 사용해서 mtcars 데이터프레임을 H2OFrame으로 변환시킨다.


~~~{.r}
class(mtcars)
~~~



~~~{.output}
[1] "data.frame"

~~~



~~~{.r}
mtcars.data <- as.h2o(mtcars)
~~~



~~~{.output}

  |                                                                       
  |                                                                 |   0%
  |                                                                       
  |=================================================================| 100%

~~~



~~~{.r}
class(mtcars.data)
~~~



~~~{.output}
[1] "H2OFrame"

~~~


## 1.7. 연습문제 {#h2o-exercise-part-1-ex07}

`h2o.dim` 함수를 사용해서 H2Oframe 차원정보를 확인한다.


~~~{.r}
h2o.dim(loan.hex)
~~~



~~~{.output}
[1] 163987     15

~~~

## 1.8. 연습문제 {#h2o-exercise-part-1-ex08}

H2Oframe 대출데이터의 칼럼명을 확인한다.


~~~{.r}
h2o.colnames(loan.hex)
~~~



~~~{.output}
 [1] "loan_amnt"             "term"                 
 [3] "int_rate"              "emp_length"           
 [5] "home_ownership"        "annual_inc"           
 [7] "purpose"               "addr_state"           
 [9] "dti"                   "delinq_2yrs"          
[11] "revol_util"            "total_acc"            
[13] "bad_loan"              "longest_credit_length"
[15] "verification_status"  

~~~

## 1.9. 연습문제 {#h2o-exercise-part-1-ex09}

H2Oframe 대출데이터에서 대출금액에 대한 히스토그램을 그려본다.


~~~{.r}
h2o.hist(loan.hex$loan_amnt)
~~~

<img src="fig/h2o-loan-histogram-1.png" style="display: block; margin: auto;" />

## 1.10. 연습문제 {#h2o-exercise-part-1-ex10}

H2Oframe 대출데이터에서 주택 소유 그룹별로 대출 평균금액을 구하시오


~~~{.r}
h2o.group_by(loan.hex, by ="home_ownership", mean("loan_amnt"))
~~~



~~~{.output}
  home_ownership mean_loan_amnt
1            ANY        5000.00
2       MORTGAGE       14586.88
3           NONE       11761.67
4          OTHER       10106.41
5            OWN       12453.96
6           RENT       11490.87

[6 rows x 2 columns] 

~~~
  
