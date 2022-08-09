# NLP competition
## General process
![](https://miro.medium.com/max/1400/1*Xw8IyDSvIoML9AMCUmvYig.png)
* load data
* Exploratory Data Analysis(EDA)
* data preprocessing
* train
* inference
* submit

## TensorFlow vs PyTorch
![](https://miro.medium.com/max/1400/1*ULcPTYyKRqF_HFTkUKdkVg.png)
![](https://www.assemblyai.com/blog/content/images/2021/12/Fraction-of-Papers-Using-PyTorch-vs.-TensorFlow.png)
![](https://www.assemblyai.com/blog/content/images/2021/12/Percentage-of-Repositories-by-Framework-----------------.png)



## HuggingFace
* Transformer 기반 pre-trained model 제공 library
* 사용자들이 model을 hub에 공유 가능
* NLP task 에서 거의 필수
* NLP process 간편하게 수행

```
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis")
>>> classifier("We are very happy to show you the 🤗 Transformers library.")

[{'label': 'POSITIVE', 'score': 0.9998}]

```

```
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
![](https://www.assemblyai.com/blog/content/images/2021/12/Number-of-Models-on-HuggingFace.png)
![](https://www.assemblyai.com/blog/content/images/2021/12/Number-of-Top-30-Models-on-HuggingFace.png)

## General strategies
* Data Preprocessing
* Data Augmentation
* Model 변경
* Hyperparameter tuning
* ensemble


### Data Imbalance strategies
* Under sampling
    * 장점 : training 시간 감소
    * 단점 : 정보 손실
    * Random Sampling
    * Tomek Links
    * CNN Rule
    * One side selection
* Over sampling
    * 장점 : 정보 손실 없음, 대체로 under sampling 보다 높은 정확도
    * 단점 : training 시간 증가, overfitting, noise나 outlier에 민감
    * Resampling
    * SMOTE(Synthetic Minority Over-sampling Technique)
    * Borderline-SMOTE
    * ADASYN(Adaptive Synthetic Sampling)


### Data Augmentation
* back translation
* [Easy Data Augmentation(EDA)](https://paperswithcode.com/paper/eda-easy-data-augmentation-techniques-for)
* [An Easier Data Augmentation(AEDA)](https://paperswithcode.com/method/aeda)


## Preferred Process
* 문제정의
* 데이터 분포, 특성 등 분석
* 문제 해결 방안 탐색
* 베이스라인 찾기(초보자)
    * task 구체화
    * HuggingFace 문서
    * kaggle, DACON 등 에서 유사한 task 코드 찾기
    * papers with code에서 논문 탐색 후 코드 찾기
* 베이스라인 모델 고정 후 다양한 실험
* hyper parameter tuning
* 실험 기록

## After competition
* 다른 참가자들과 공유 하고 분석
* 좋은 결과, 그렇지 못한 결과 원인 분석
* 추후에 시도해 볼 것들
* 자세히 정리
* 코드 리팩토링
* 공식문서 자세히 보기
* 잘 모르고 시도했던 실험들 개념, 장단점, 특징 자세히 알아보기
* 프레임워크, 라이브러리 소스코드 보고 동작방식 살펴보기
* 다음대회 도전
* 반복


## NLP Competitions
* DACON, kaggle 이외에도 많은 대회가 열리는 추세
* 초보자는 DACON 추천(분류 문제가 시작하기 쉽고 자료도 많음)
    * 최근 대회의 경우 상위권 코드 공유 활발
    * 상대적으로 낮은 허들
    * 한국어
        * 익숙
        * 자료가 상대적으로 적음, 자료가 많은 영어와 다른 언어적 특성(형태소, 조사, 띄어쓰기 등)
    * HuggingFace는 NLP에서 거의 필수(+PyTorch)

## NLP study
![](https://github.com/graykode/nlp-roadmap/blob/master/img/nlp.png?raw=true)
* [초보자 추천교재](https://wikidocs.net/book/2155)
* [HuggingFace](https://huggingface.co/)
* [Paperswithcode](https://paperswithcode.com/)
* 

## Tools
* [Weights and Biases](https://wandb.ai/home)
* [OPTUNA](https://optuna.org/)

## NLP data
* 공개 데이터
    * [AI-HUB](https://www.aihub.or.kr/aihubdata/extrlpltfomdata/list.do?currMenu=118&topMenu=100)
    * [모두의 말뭉치](https://corpus.korean.go.kr/)
    * [Exobrain](http://exobrain.kr/pages/ko/result/outputs.jsp)
* 직접 수집
    * 크롤링
        * 저작권 이슈에 주의


## For Vision or ML
* [머신러닝 문제 해결 전략(Vision, ML)](http://www.yes24.com/Product/Goods/108802734)
* [머신러닝 문제 해결 전략 github](https://github.com/BaekKyunShin/musthave_mldl_problem_solving_strategy)
* [머신러닝 마스터 클래스(ML)](http://www.yes24.com/Product/Goods/97559803)


## Reference

- [https://campaign.naver.com/clova_airush/](https://campaign.naver.com/clova_airush/)
- [https://gurukannan.medium.com/overview-of-mlops-ml-dev-ops-2899ecb97820](https://gurukannan.medium.com/overview-of-mlops-ml-dev-ops-2899ecb97820)
- [https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2022/](https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2022/)
- [https://medium.com/analytics-vidhya/ml03-9de2f0dbd62d](https://medium.com/analytics-vidhya/ml03-9de2f0dbd62d)
- [https://github.com/graykode/nlp-roadmap](https://github.com/graykode/nlp-roadmap)

