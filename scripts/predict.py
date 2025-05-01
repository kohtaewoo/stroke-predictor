import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image
import os

# --- 경로 설정 ---
MODEL_PATH = "model/mlp_composite_model.joblib"
SCALER_PATH = "model/scaler_composite.joblib"
DATA_DIR = "data"

# --- 모델 및 스케일러 불러오기 ---
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- Mediapipe 초기화 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# --- 좌표 정의 ---
symmetry_pairs = [(61, 291), (78, 308), (76, 306)]  # 입꼬리, 윗입술, 입술 중간
LEFT_EYE, RIGHT_EYE, NOSE_TIP = 33, 263, 1
data = []

# --- Yaw 계산 ---
def estimate_yaw(landmarks, image_size):
    image_w, image_h = image_size
    indices = [1, 33, 263, 61, 291, 199]
    model_points = np.array([
        [0.0, 0.0, 0.0], [-30.0, -30.0, -30.0], [30.0, -30.0, -30.0],
        [-30.0, 30.0, -30.0], [30.0, 30.0, -30.0], [0.0, 50.0, -10.0]
    ], dtype=np.float32)
    image_points = [(int(landmarks[idx].x * image_w), int(landmarks[idx].y * image_h)) for idx in indices]
    image_points = np.array(image_points, dtype=np.float32)
    focal_length = image_w
    center = (image_w / 2, image_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
    yaw = np.degrees(np.arctan2(-rotation_mat[2, 0], sy))
    return yaw

# --- Roll 계산 ---
def estimate_roll(landmarks):
    dx = landmarks[RIGHT_EYE].x - landmarks[LEFT_EYE].x
    dy = landmarks[RIGHT_EYE].y - landmarks[LEFT_EYE].y
    return np.degrees(np.arctan2(dy, dx))

# --- 특징 추출 ---
def extract_features(landmarks):
    eye_dist = np.linalg.norm(
        np.array([landmarks[LEFT_EYE].x, landmarks[LEFT_EYE].y]) -
        np.array([landmarks[RIGHT_EYE].x, landmarks[RIGHT_EYE].y])
    )
    features = []
    for l_idx, r_idx in symmetry_pairs:
        lx, ly = landmarks[l_idx].x, landmarks[l_idx].y
        rx, ry = landmarks[r_idx].x, landmarks[r_idx].y
        dist = np.linalg.norm(np.array([lx, ly]) - np.array([rx, ry]))
        dx = lx - rx
        dy = ly - ry
        nx_l = (lx - landmarks[NOSE_TIP].x) / (eye_dist + 1e-6)
        ny_l = (ly - landmarks[NOSE_TIP].y) / (eye_dist + 1e-6)
        nx_r = (rx - landmarks[NOSE_TIP].x) / (eye_dist + 1e-6)
        ny_r = (ry - landmarks[NOSE_TIP].y) / (eye_dist + 1e-6)
        features.extend([dist, dx, dy, nx_l, ny_l, nx_r, ny_r])
    return features

# --- 한글 출력 함수 ---
def draw_text(img, text, pos=(30, 40), size=26, color=(255, 255, 255)):
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", size)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- 웹캠 실행 ---
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    msg, color = "Y: 정상 저장 / N: 비정상 저장", (255, 255, 255)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # 입 좌표 시각화
        for idx in [i for pair in symmetry_pairs for i in pair]:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
            cv2.putText(frame, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        yaw = estimate_yaw(landmarks, (w, h))
        roll = estimate_roll(landmarks)

        if abs(yaw) > 10 or abs(roll) > 14:
            msg = f"(Yaw: {yaw:.1f}°, Roll: {roll:.1f}°) → 정면 아님"
            color = (180, 180, 180)
            features = None
        else:
            features = extract_features(landmarks)
            features.append(yaw)
            features.append(roll)
            features_arr = np.array(features).reshape(1, -1)

            # ✅ 경고 제거: 컬럼 이름 포함
            columns = scaler.feature_names_in_
            features_df = pd.DataFrame(features_arr, columns=columns)

            features_scaled = scaler.transform(features_df)
            pred = model.predict(features_scaled)[0]

            if pred == 0:
                msg = "✅ 예측: 정상 (Y로 저장)"
                color = (0, 255, 0)
            else:
                msg = "❌ 예측: 비정상 (N으로 저장)"
                color = (0, 0, 255)
    else:
        features = None
        msg = "얼굴 인식 실패"
        color = (120, 120, 120)

    frame = draw_text(frame, msg, color=color)
    cv2.imshow("🧠 실시간 예측 + 수동 피드백 수집", frame)

    key = cv2.waitKey(5)
    if key == 27:
        break
    elif key == ord('y') and features:
        data.append(features + [0])
        print("✅ 정상 샘플 저장")
    elif key == ord('n') and features:
        data.append(features + [1])
        print("❌ 비정상 샘플 저장")

cap.release()
cv2.destroyAllWindows()

# --- CSV 저장 ---
n = len(symmetry_pairs)
columns = [f"dist_{i}" for i in range(n)] + [f"dx_{i}" for i in range(n)] + [f"dy_{i}" for i in range(n)] + \
          [f"norm_lx_{i}" for i in range(n)] + [f"norm_ly_{i}" for i in range(n)] + \
          [f"norm_rx_{i}" for i in range(n)] + [f"norm_ry_{i}" for i in range(n)] + \
          ["yaw", "roll", "label"]

if data:
    df = pd.DataFrame(data, columns=columns)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"{DATA_DIR}/composite_feature_data_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"💾 저장 완료: {filename}")
else:
    print("⚠️ 저장할 데이터 없음.")
