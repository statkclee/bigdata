---
layout: page
title: 빅데이터
subtitle: 분석할 데이터를 스파크에 적재하는 방법 - S3 포함
author:
    name: xwMOOC
    url: https://www.facebook.com/groups/tidyverse/
    affiliation: Tidyverse Korea
date: "2019-01-18"
output:
  html_document: 
    toc: yes
    toc_float: true
    highlight: tango
    code_folding: show
    number_section: true
    self_contained: true
editor_options: 
  chunk_output_type: console
---






## AWS S3 데이터를 EC2 RStudio에서 읽어오기 [^aws-s3-read-write] [^cloudyr-aws-s3] {#load-data-on-spark-read}

[^aws-s3-read-write]: [Read and Write Data To and From Amazon S3 Buckets in Rstudio](http://datascience.ibm.com/blog/read-and-write-data-to-and-from-amazon-s3-buckets-in-rstudio/)

[^cloudyr-aws-s3]: [Amazon Simple Storage Service (S3) API Client](https://github.com/cloudyr/aws.s3)


EC2에 RStudio 서버가 설치되면 AWS S3 저장소의 데이터를 불러와서 작업을 해야한다.

<img src="fig/aws-ec2-s3-link.png" alt="AWS EC2 S3 연결" width="67%" />

## 환경설정 {#load-spark-strategy-setup}

`aws.s3` 팩키지를 통해 AWS S3와 R이 작업을 할 수 있도록 한다. `devtools::install_github("cloudyr/aws.s3")` 명령어를 통해 팩키지를 설치한다.
`Sys.setenv` 명령어를 통해 `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` 을 설정한다.



```r
# aws.s3 설치
devtools::install_github("cloudyr/aws.s3")

library(aws.s3)

# S3 버킷 접근을 위한 키값 설정
Sys.setenv("AWS_ACCESS_KEY_ID" = "xxx",
           "AWS_SECRET_ACCESS_KEY" = "xxx",
           "AWS_DEFAULT_REGION" = "ap-northeast-2")
```

## 설정환경 확인 {#load-spark-strategy-setup-check}

`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`에 부여된 권한에 맞춰 제대로 S3 버킷에 접근할 수 있는지 확인한다.
접근 권한이 있는 모든 S3 버킷 정보가 화면에 출력된다.



```r
bucketlist()
```

## `S3` 버킷 헬로 월드 {#load-data-on-spark-strategy-run}

`get_object` 명령어를 통해 `v-seoul` 버킷 최상위 디렉토리에 있는 `iris.csv` 파일을 불러온다.
바이러니 형태라 사람이 읽을 수 있는 문자형으로 변환시키고 `textConnection` 함수를 통해 `.csv` 파일을 
R에서 작업할 수 있는 데이터프레임 형태로 변환시킨다.


```r
iris_dat <- get_object("iris.csv", bucket = "v-seoul")

iris_obj <- rawToChar(iris_dat)  
con <- textConnection(iris_obj)  
iris_df <- read.csv(con)  
close(con)  

iris_df
```

# AWS S3 데이터를 스파크 EMR 클러스터 RStudio에서 읽어오기 {#load-data-on-spark-strategy-cluster}

AWS EMR 클러스터를 통해 S3에 저장된 대용량 데이터 특히 `.parquet` 형태로 압축된 데이터인 경우, 
우선 다음 작업과정을 거쳐 분석 가능한 형태 데이터로 정제한다.

1. `.parquet` 데이터를 `spark_read_parquet` 함수로 스파크 데이터프레임으로 읽어온다.
1. `sparklyr` 팩키지 `sdf_sample()` 함수를 활용하여 표본 추출한다. 
    - 1억건이 넘어가는 데이터의 경우 100 GB를 쉽게 넘고 0.1% 표본추출해도 수백MB가 된다.
1. `collect()` 함수를 통해 스파크 데이터프레임을 R 데이터프레임으로 변환한다.
1. `aws.s3` 팩키지 `s3save()` 함수로 인메모리 S3객체를 `.Rdata` 파일로 S3 버킷에 저장시킨다.
    - 상기 과정을 `save_s3_from_parquet` 함수로 만들어 활용한 사례가 아래 나와 있다.
1. 마지막으로 `aws.s3` 팩키지 `s3load()` 함수를 통해 EC2 컴퓨터에서 불러와서 R로 후속 작업을 이어간다.

<img src="fig/aws-s3-dataframe-sync.png" alt="AWS EC2 스파크 S3 연결" width="57%" />



```r
save_s3_from_parquet <- function(spark_df, parquet_file, frac) {
  tmp_df <- spark_read_parquet(sc, paste0("df_", spark_df), parquet_file)
  tmp_df_smpl <- sdf_sample(tmp_df, fraction = frac, replacement = FALSE, seed = NULL)
  tmp_df_smpl_df <- collect(tmp_df_smpl)
  s3save(tmp_df_smpl_df, bucket = "S3버킷명/경로명", object = paste0(spark_df, "_smpl_df.Rdata"))
}

save_s3_from_parquet("df_0325", "s3://버킷명/2017/03/action_20170326.parquet", 0.01)
```

