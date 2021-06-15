# Latex Mathematics Expression Recognition

## Members.

[김익재](https://github.com/ijjustinKim) | [남지인](https://github.com/zeen263) | [이주남](https://github.com/joon1230) | [이진성](https://github.com/ssun-g) | [장형주](https://github.com/hangjoo) | [최길희](https://github.com/grazerhee)

## Main Features.

- 실험 커스터마이징을 위해 코드를 수정할 필요 없이 최대한 config yaml 내에서 수정하는 것으로 가능하도록 구현.
- wandb 연동으로 실험 관리
- 이미지를 transform 시 성능 향상을 위해 albumentation transform 사용.
- 다양한 환경 설정을 위해 코드를 모듈화해서 분리해서 사용. Main Code는 train.py를 통해 모델을 학습하고 core 디렉토리에서 조립하여 학습 환경에 적용할 수 있도록 코드 구현.

## Quick Start

1. 패키지 설치
```
pip install -r requirements.txt
```
2. 패키지 설치 후 configs 디렉토리에서 알맞은 모델 config을 선택한 뒤, 실험 환경에 맞게 인자를 설정(아래 Config.yaml 참고)
3. config 내에 설정한 데이터 경로에 맞게 데이터를 이동.
4. train.py 실행
```bash
python ./train.py -c ./config/SATRN.yaml
```

## Config.yaml

```yaml
model:
  type: "Baseline_SATRN"
  encoder:
    hidden_dim: 300
    filter_dim: 600
    layer_num: 6
    head_num: 8
  decoder:
    src_dim: 300
    hidden_dim: 128
    filter_dim: 512
    layer_num: 3
    head_num: 8
  dropout_rate: 0.1
loss:
  type: "CrossEntropyLoss"
optimizer:
  type: "Adam"
  lr: 5e-4
  weight_decay: 1e-4
scheduler:
  type: "CircularLRBeta"
  lr_max: 5e-4
  lr_divider: 10
  cut_point: 10
  step_size: 834  # step size
  momentum:
    - 0.95
    - 0.85

data:
  train:
    path:
      - "{path where gt.txt located for training.}"
    transforms:
      - "Resize(height=32, width=100)"
      - "Normalize(always_apply=True)"
  valid:
    path:
      - ""
    transforms:
      - null
  test:
    path:
      - ""
    transforms:
      - null
  token_paths:
    - "{Path where token file located.}"
  dataset_proportions:
    - 1.0
  random_split: True
  test_proportions: 0.2
  rgb: True

train_config:
  batch_size: 64
  num_workers: 8
  num_epochs: 50
  print_interval: 1
  teacher_forcing_ratio: 0.5
  max_grad_norm: 2.0
  fp_16: False
seed: 42
wandb:
  project: "Project name"
  entity: "Group you joined"
  name: "Experiment name you want"
prefix: "./log/SATRN"
checkpoint: ""
```

### model

- type : 학습에 사용할 모델
- others : model 생성시 arguments로 전달될 인자

### loss

- type : 학습에 사용될 loss
- others : Loss function 생성시 arguments로 전달될 인자
- ignore_index 인자로 자동으로 Tokenizer의 PAD Token index 설정.

### optimizer

- type : 학습에 사용될 optimizer
- others : optimizer 생성시 arguments로 전달될 인자

### scheduler

- type : 학습에 사용될 scheduler
- others : scheduler 생성시 arguments로 전달될 인자

### data

- train
    - path : 학습에 사용될 ground truth text가 저장된 경로. 해당 파일은 줄마다 이미지 파일명과 해당 이미지에 Ground Truth에 해당하는 Latex 문자열이 기호마다 띄어쓰기(" ")로 구분되어 있는 파일이여야 하며, 이미지 파일은 같은 경로 내에 images 디렉토리 내에 위치해야함.
    - transforms : 이미지에 적용할 transform 리스트. albumentation 사용하듯 동일하게 문자열로 전달.

        ```python
        # transforms:
        #     - "Resize(height=128, width=128)"
        #     - "Normalize(always_apply=True)"
        # 위의 transfomrs는 아래와 같이 변경되어 적용됨.

        transforms = A.Compose([
        	A.Resize(height=128, width=128),
        	A.Normalize(always_apply=True),
        ])
        ```

- valid
    - path : 검증에 사용될 ground truth text가 저장된 경로. train 인자와 동일하게 파일과 경로가 세팅되어야 함.
    - transforms : 이미지에 적용할 transform 리스트. train과 동일하게 사용.
- token_paths : 학습에 사용될 토큰들이 저장된 txt 파일 경로. txt 파일은 각 토큰들이 개행 문자("\n")로 나뉜 txt 파일이어야 함.
- dataset_proportions : train 인자의 각 path에서 어느정도 비율만 가져올지 설정하는 인자. dataset_proportions가 0.7인 경우 train 데이터에서 랜덤으로 0.7만큼만 가져와 학습에 사용함.
- random_split : 해당 인자가 True일 경우 valid 인자는 무시되고 train 데이터에서 test_proportions만큼 train과 valid 데이터를 나눠서 각각 학습과 검증에 사용함. transform도 동일하게 train 인자로 전달된 transform을 validation에 동일하게 적용.
- test_proportions : train 데이터에서 validation에 사용할 데이터의 비율을 지정하는 인자. random_split이 True인 경우 같이 설정해주어야 함.
- rgb : True인 경우 3채널 RGB, False인 경우 1채널 Grayscale로 이미지를 불러옴. Grayscale로 이미지를 불러오는 경우 적용안되는 transform들(ex. Normalize)이 다수 있으므로 사용 시 유의.

### train_config

- batch_size : 배치 사이즈
- num_workers : 데이터를 로딩하는데 사용할 서브 프로세스의 수
- num_epochs : 학습 epoch 수
- print_interval : log를 기록할 interval 인자
- dropout_rate : dropout 인자
- teacher_forcing_ratio : teacher_forcing 인자
- max_grad_norm : gradient clipping 적용시 최대 gradient 값

### etc

- seed : 고정될 seed 값
- wadb : wandb.init 시 사용할 인자
- prefix : log 파일과 checkpoint 파일을 저장할 디렉토리 경로
- checkpoint : checkpoint 파일(.pth) 경로가 주어질 경우 해당 checkpoint 파일을 불러와 학습을 해당 checkpoint에서부터 진행