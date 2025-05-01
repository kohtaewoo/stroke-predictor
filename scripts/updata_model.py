import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

# --- 경로 설정 ---
DATA_DIR = "data"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_composite_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_composite.joblib")

# --- CSV 파일 불러오기 ---
files = sorted(glob.glob(f"{DATA_DIR}/composite_feature_data_*.csv"))
if not files:
    raise FileNotFoundError("🛑 CSV 파일이 없습니다. 먼저 데이터를 수집하세요.")

df_list = []
for f in files:
    temp = pd.read_csv(f)
    if "label" in temp.columns and temp["label"].notna().all():
        df_list.append(temp)

if not df_list:
    raise ValueError("❌ 유효한 데이터가 없습니다.")

df = pd.concat(df_list, ignore_index=True)
print(f"📂 불러온 파일 수: {len(files)}개, 총 데이터 수: {len(df)}")

# --- 특징과 라벨 분리 ---
X = df.drop("label", axis=1)
y = df["label"].astype(int)

if y.value_counts().min() < 5:
    print("⚠️ 일부 클래스 데이터 수가 너무 적습니다.")

# --- 정규화 ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 데이터 분할 ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# --- 모델 학습 ---
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
model.fit(X_train, y_train)

# --- 평가 ---
y_pred = model.predict(X_test)
print("📊 분류 성능:")
print(classification_report(y_test, y_pred))

# --- 모델 저장 ---
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"✅ 저장 완료: {MODEL_PATH}, {SCALER_PATH}")
