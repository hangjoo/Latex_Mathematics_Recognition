<br></br>
### Dataset 전략
<br></br>
**train data ( 8만장 )**

- train data.txt : 군집화를 통해서 계층화 추출을 한 (8만 - train_edge_case)장의 이미지 파일 이름들을 모아두었습니다.
- train_edge_case.json : 50개 미만의 희소한 토큰들을 포함한 이미지. edge case 로 inference 결과를 봐야할 데이터지만 학습엔 반드시 포함하는 데이터 입니다. json 형태로 되어 있으며 각 token당  해당 이미지를 담고있는 형태입니다.

**validation ( 2만장 )**

- validation data.txt : 군집화를 통해서 계층호 추출을 한 ( 2만 - validation_edge_case ) 장의 이미지 입니다.
- validation_edge_case.json : inference 결과를 보고 싶은 이미지 입니다. json 형태로 구성되어 있고 아래와 같이 분류되어 있습니다.
    - 궁금한 토큰들 ('\\end{matrix}','\\begin{matrix}','\\sum') 3장 씩
    - 50 ~300 개의 token을 가지고 있는 이미지들 ( 각 토큰별 적어도 3장 )
    - 흐린 이미지 20 장
    - 세로/가로 비율이 (10배 이상인, 0.5배 미만인) 인 이미지 각각 30장