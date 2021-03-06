---
layout: page
title: 빅데이터
---

> ### AI is a Superpower {.callout}
>
> "AI is a superpower!!!", 인공지능을 체득하면 슈퍼파워를 손에 쥘 것이다. [Andrew Ng](https://twitter.com/andrewyng/status/728986380638916609)
>
> 금수저, 은수저 슈퍼파워를 받은 사람과 기계학습을 통달한 흑수저들간의 무한경쟁이 드뎌 시작되었다. 물론, 
> 금수저를 입에 물고 기계학습을 통달한 사람이 가장 유리한 출발을 시작한 것도 사실이다.


## 학습목차 

- [빅데이터 스파크(Spark)](bigdata-spark-mooc.html)
    + [빅데이터 들어가며 - 기본기](bigdata-pyspark-prerequisite.html)
    + [시나리오별 클라우드(AWS) 컴퓨팅 자원을 데이터 과학에 활용](ml-aws-from-pc.html)
    + [스파크 로컬 클러스터 - 시운전(Dry Run)](bigdata-local-sparklyr.html)
- [파이스파크 - `pyspark`](bigdata-pyspark.html)
    * [데이터(S3) 읽어오기 - 파이썬](bigdata-pyspark-read-data.html)            
    * [자료 구조](bigdata-pyspark-data-structure.html)
    * [자료 변환](bigdata-pyspark-data-transformation.html)
    * [데이터프레임과 SQL](bigdata-pyspark-dataframe-sql.html)
    * [기계학습 예측모형](bigdata-pyspark-predictive-model.html) (TBA)
    * [스파크 SQL(Spark SQL)](bigdata-spark-sql.html)
- RStudio 스파크 - `sparklyr`
    * **스파크 설치**
        - [로컬 컴퓨터 스파크 설치](ds-sparklyr.html) 
        - [AWS EC2 스파크 설치](ds-install-sparklyr-ec2.html) 
    - [데이터(S3) 읽어오는 전략 - R](bigdata-sparklyr-read-data.html)
        + [로컬 `.csv` &rarr; 로컬 스파크 - `sparklyr`](bigdata-sparklyr-read-data-csv-local.html)
        + [S3 &rarr; 로컬 RStudio IDE](bigdata-sparklyr-read-data-s3-local.html)
        + [S3 &rarr; EC2 RStudio 서버](bigdata-sparklyr-read-data-s3-ec2.html)
        + [S3 &rarr; EC2 서버 - 스파크](bigdata-sparklyr-read-data-s3-spark.html)
        + [분석할 데이터를 스파크에 적재하는 방법 - S3 포함](ml-ec2-s3.html)
    * [Spark와 연결하는 `sparklyr`, `dplyr`, 그리고 기계학습](ml-sparklyr.html)
- [$H_2 O$ 기계학습](h20-arch.html)
    + [$H_2 O$ 하둡 스파크 클러스터 설치](ds-h2o-spark-hadoop.html)    
    + [$H_2 O$ R 연습문제](h2o-r-exercise.html)
    + [$H_2 O$ 앙상블 모형](h2o-ensemble-higgs.html)
    + [$H_2 O$ GBM 모형 세부조정](h2o-gbm-titanic.html)
- [기계학습 클라우드(AWS) 개발배포 환경](ml-aws-spark.html) (TBA)    
    + [AWS 우분투 EC2 + S3 버킷 + RStudio 서버](ds-aws-rstudio-server.html)
    + [AWS EMR - `sparklyr`](bigdata-aws-emr.html)    
    + [스파크 EC2 클러스터 - 부싯돌(flintrock)](ml-aws-ec2-flintrock.html)
        - [스파크 EC2 클러스터 - 데이터과학 툴체인(R, sparklyr)](ml-aws-ec2-flintrock-sparklyr.html)
- **AWS Boto3 - 파이썬**
    - [AWS IAM](cloud-aws-iam.html)
    - [정적 웹호스팅 - `S3`](cloud-aws-s3-web-hosting.html)
    - [개인 도메인 블로그 - `aws cli`](cloud-aws-s3-blog.html)
- **참고**
    - [빅메모리(bigmemory)](bigdata-bigmemeory.html): "빅데이터는 디스크에 쓰고 R 메모리라고 읽는다"
    - [SparkR 들어가며](sparkr-intro.html)
        - [SparkR 설치](spark-hadoop-install.html)
        - [SparkR 헬로 월드](sparkr-hello-world.html)
        - [SparkR 도커](sparkr-docker.html)
        - [우분투 SparkR 설치](sparkr-ubuntu.html)
        - [SparkR 하둡 클러스터 설치](ds-spark-hadoop-install.html)
    - [EMR 스파크 클러스터 - wadal](ml-emr-wadal.html)
    - 웹 크롤링(Crawling): [데이터 크롤링 &rarr; S3](ml-crawling-s3.html)
    - 웹 크롤링(Crawling): [데이터 크롤링: EC2 &rarr; S3](ml-crawling-ec2-s3.html)

### [xwMOOC 오픈 교재](https://statkclee.github.io/xwMOOC/)

- **컴퓨팅 사고력(Computational Thinking)**
    - [컴퓨터 과학 언플러그드](http://statkclee.github.io/unplugged)  
    - [리보그 - 프로그래밍과 문제해결](https://statkclee.github.io/code-perspectives/)  
         - [러플](http://statkclee.github.io/rur-ple/)  
    - [파이썬 거북이](http://swcarpentry.github.io/python-novice-turtles/index-kr.html)  
    - [정보과학을 위한 파이썬](https://statkclee.github.io/pythonlearn-kr/)  
    - [소프트웨어 카펜트리 5.3](http://statkclee.github.io/swcarpentry-version-5-3-new/)
    - [기호 수학(Symbolic Math)](https://statkclee.github.io/symbolic-math/)
    - [데이터 과학을 위한 R 알고리즘](https://statkclee.github.io/r-algorithm/)
    - [데이터 과학을 위한 저작도구](https://statkclee.github.io/ds-authoring/)
        - [The Official xwMOOC Blog](https://xwmooc.netlify.com/)
    - [비즈니스를 위한 오픈 소스 소프트웨어](http://statkclee.github.io/open-source-for-business/)
- **데이터 과학**
    - [R 데이터과학](https://statkclee.github.io/data-science/)
    - [시각화](https://statkclee.github.io/viz/)
    - [텍스트 - 자연어처리(NLP)](https://statkclee.github.io/text/)
    - [네트워크(network)](https://statkclee.github.io/network)
    - [데이터 과학– 기초 통계](https://statkclee.github.io/statistics/)    
        - [공개 기초 통계학 - OpenIntro Statistics](https://statkclee.github.io/openIntro-statistics-bookdown/)
    - [공간통계를 위한 데이터 과학](https://statkclee.github.io/spatial/)
    - [~~R 팩키지~~](http://r-pkgs.xwmooc.org/)
    - [~~통계적 사고~~](http://think-stat.xwmooc.org/)
    - [보안 R](https://statkclee.github.io/security/) - TBA
- **빅데이터**
    - [빅데이터(Big Data)](http://statkclee.github.io/bigdata)
    - [데이터 제품](https://statkclee.github.io/data-product/)
    - [R 도커](http://statkclee.github.io/r-docker/)
- **기계학습, 딥러닝, 인공지능**
    - [기계학습](http://statkclee.github.io/ml)
    - [딥러닝](http://statkclee.github.io/deep-learning)
    - [R 병렬 프로그래밍](http://statkclee.github.io/parallel-r/)
    - [고생대 프로젝트](http://statkclee.github.io/trilobite)
    - [인공지능 연구회](https://statkclee.github.io/ai-lab/)
- [IoT 오픈 하드웨어(라즈베리 파이)](http://statkclee.github.io/raspberry-pi)
    - [$100 오픈 컴퓨터](https://statkclee.github.io/one-page/)   
    - [$100 오픈 슈퍼컴퓨터](https://statkclee.github.io/hpc/)
- [선거와 투표](http://statkclee.github.io/politics)
    - [저녁이 있는 삶과 새판짜기 - 제7공화국](https://statkclee.github.io/hq/)



