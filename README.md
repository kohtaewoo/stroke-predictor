# 🧠 Stroke Predictor (FAST 기반 뇌졸중 예측 시스템)

> 웹캠을 통해 얼굴을 실시간으로 인식하고, 비대칭 여부, Yaw/Roll 등 얼굴 특징을 기반으로  
> **뇌졸중 징후(얼굴 마비 등)**를 탐지하여 정상/비정상을 예측하는 AI 모델입니다.

---

## 📦 프로젝트 구성

```
stroke-predictor/
├── scripts/
│   ├── predict.py             # 실시간 예측 및 수동 데이터 수집 스크립트 (웹캠 기반)
│   └── updata_model.py        # 모델 업데이트용 스크립트 (선택사항)
├── data/                      # 수집된 특징 CSV 저장 폴더
├── model/
│   ├── mlp_composite_model.joblib   # 학습된 MLP 모델
│   └── scaler_composite.joblib      # StandardScaler 객체
├── .gitignore
└── README.md
```

---

## 🚀 실행 방법

### 1. 설치

```bash
pip install -r requirements.txt
```

> ※ `requirements.txt` 예시:
> ```
> opencv-python
> mediapipe
> pandas
> numpy
> joblib
> Pillow
> ```

### 2. 실시간 예측 실행

```bash
python scripts/predict.py
```

실행 후 웹캠이 열리며, 얼굴이 인식되면 예측 결과가 화면 상단에 나타납니다.

---

## ✅ 기능 설명

- ✅ Mediapipe FaceMesh 기반 **얼굴 랜드마크 추출**
- ✅ Yaw, Roll 값 기반 정면 판별
- ✅ **입 좌우 대칭 비율** 및 정규화된 거리 계산
- ✅ 학습된 MLP 모델을 통해 **정상/비정상 판단**
- ✅ 키보드 입력을 통한 라벨링 수집 (`Y`: 정상, `N`: 비정상)
- ✅ 수집된 데이터는 `data/composite_feature_data_*.csv` 형식으로 저장됨

---

## 🧪 특징 추출 항목

- 좌우 대칭 좌표쌍 기반 거리 / 차이 (dx, dy)
- 각 좌표의 코를 기준으로 한 정규화 위치
- Yaw (좌우 회전), Roll (기울기) 각도
- 총 `n * 7 + 2`개의 특징 + 라벨

---

## 📸 사용 방법 요약

| 상황             | 동작                                                    |
|------------------|---------------------------------------------------------|
| 얼굴 인식 실패   | `얼굴 인식 실패` 메시지 출력                              |
| 정면이 아닐 경우 | `(Yaw, Roll) → 정면 아님` 회색 메시지 출력                |
| 정상 예측        | `✅ 예측: 정상 (Y로 저장)` 녹색 메시지 출력                |
| 비정상 예측      | `❌ 예측: 비정상 (N으로 저장)` 빨간 메시지 출력            |
| 키 입력 `Y`       | 현재 특징을 라벨 `0`(정상)으로 저장                        |
| 키 입력 `N`       | 현재 특징을 라벨 `1`(비정상)으로 저장                      |

---

## 💾 저장 파일 구조

```csv
dist_0, ..., dx_0, dy_0, norm_lx_0, ..., yaw, roll, label
0.032, ..., ..., ..., ..., ..., ..., ..., 1
```

---

## 📌 참고 사항

- `malgun.ttf` 경로는 Windows 기준이며, macOS/Linux에서는 다른 폰트로 교체 필요
- Mediapipe는 CPU 성능에 따라 딜레이가 발생할 수 있음
- WebRTC 또는 모바일 이식 가능성 있음 (추후 확장)

---

## 👨‍💻 제작자

- GitHub: [kohtaewoo](https://github.com/kohtaewoo)
- 프로젝트명: **Stroke FAST Predictor**
- 설명: 얼굴 대칭과 시선 정보를 기반으로 뇌졸중 전조증상 예측

