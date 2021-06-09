# Latex Mathematics Expression Recognition

## What diff with baseline.

- train.py 수정 필요 없이 최대한 config.yaml 내에서 커스터마이징 되도록 구현.
- wandb 연동
- pytorch transforms에서 albumentation transforms로 변경.
- config.yaml 내에 loss, optimizer, scheduler 설정 추가.

## Config.yaml

```yaml
model:
  type: "Baseline_Attention"
  Attention:
    src_dim: 512
    hidden_dim: 128
    embedding_dim: 128
    layer_num: 1
    cell_type: "LSTM"
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
  step_size: 834
  momentum:
    - 0.95
    - 0.85

data:
  train:
    path:
      - "/opt/ml/input/data/train_dataset/gt.txt"
    transforms:
      - "Resize(height=128, width=128)"
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
    - "/opt/ml/input/data/train_dataset/tokens.txt"
  dataset_proportions:
    - 1.0
  random_split: True
  test_proportions: 0.2
  crop: True
  rgb: True

train_config:
  batch_size: 96
  num_workers: 8
  num_epochs: 50
  print_interval: 1
  dropout_rate: 0.1
  teacher_forcing_ratio: 0.5
  max_grad_norm: 2.0
seed: 42
wandb:
  project: "hangjoo"
  entity: "unnamed"
  name: "Attention_baseline"
prefix: "./log/attention_50"
checkpoint: ""
```

### model

- type : 학습에 사용할 모델
- others : model 생성시 arguments로 전달될 인자

### loss

- type : 학습에 사용될 loss
- others : Loss function 생성시 arguments로 전달될 인자

### optimizer

- type : 학습에 사용될 optimizer
- others : optimizer 생성시 arguments로 전달될 인자

### scheduler

- type : 학습에 사용될 scheduler
- others : scheduler 생성시 arguments로 전달될 인자

### data

- train
    - path : 학습에 사용될 ground truth text가 저장된 경로
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
    - path : 검증에 사용될 ground truth text가 저장된 경로
    - transforms : 이미지에 적용할 transform 리스트. train과 동일하게 사용.
- token_paths : 학습에 사용될 토큰들이 저장된 txt 파일 경로. txt 파일은 각 토큰들이 개행 문자("\n")로 나뉜 txt 파일이어야 함.
- dataset_proportions : train 인자의 각 path에서 어느정도 비율만 가져올지 설정하는 인자. dataset_proportions가 0.7인 경우 train 데이터에서 랜덤으로 0.7만큼만 가져와 학습에 사용함.
- random_split : 해당 인자가 True일 경우 valid 인자는 무시되고 train 데이터에서 test_proportions만큼 train과 valid 데이터를 나눠서 각각 학습과 검증에 사용함.
- test_proportions : train 데이터에서 validation에 사용할 데이터의 비율을 지정하는 인자.
- crop : 사용하지 않음.
- rgb : True인 경우 RGB, False인 경우 Grayscale로 이미지를 불러옴. 채널 수도 각각 3채널, 1채널로 읽어옴.

### train_config

- batch_size : 배치 사이즈
- num_workers :
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

## TODO

- [x]  ~~Wandb 연동~~
- [x]  ~~Wandb에 config.yaml, checkpoint.pth 파일 업로드~~
- [x]  ~~Epoch / Step Scheduler 구별~~
- [x]  ~~token_to_id, id_to_token → Tokenizer 클래스로 분리해서 사용 → 프로젝트 때 API나 Inference 시에 사용해야하므로 dataset에 종속되게 구현하는 것보다 따로 떼어내서 사용하는 것이 용이함.~~
- [x]  ~~Albumentation transfrom 구현~~
- [ ]  loss function → ignore index 추가 → id_to_string 함수 수정 필요
- [x]  ~~PAD 토큰을 -1로 변환하는 코드(355라인 in train.py) 제거 → id_to_string에서는 PAD 토큰이 무엇인지 몰라서 -1로 변환~~
- [ ]  run_epoch 코드 수정
- [x]  ~~Inference 코드도 맞게 수정~~
- [ ]  API 코드 구현
- [ ]  SATRN 모델 구현