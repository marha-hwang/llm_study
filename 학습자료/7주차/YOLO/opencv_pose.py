import cv2, time, math
import numpy as np
from ultralytics import YOLO

# 1) 포즈 모델 로드 (COCO 17 keypoints)
model = YOLO("yolo11n-pose.pt")
# model.to("cuda")  # GPU가 있으면 활성화

# 2) 스켈레톤 연결 규칙(COCO)
SKELETON = [
    (5, 6),   # L-Shoulder — R-Shoulder
    (5, 7), (7, 9),     # L-Shoulder — L-Elbow — L-Wrist
    (6, 8), (8, 10),    # R-Shoulder — R-Elbow — R-Wrist
    (5, 11), (6, 12), (11, 12),  # 어깨-엉덩이 연결
    (11, 13), (13, 15),  # Left Hip — Knee — Ankle
    (12, 14), (14, 16),  # Right Hip — Knee — Ankle
    (0, 1), (0, 2), (1, 3), (2, 4),  # 얼굴 주변
    (0, 5), (0, 6)  # Nose — Shoulders
]
# 키포인트 인덱스 참고: 0코,1왼눈,2오른눈,3왼귀,4오른귀,5왼어깨,6오른어깨,7왼팔꿈치,8오른팔꿈치,9왼손목,10오른손목,11왼엉덩이,12오른엉덩이,13왼무릎,14오른무릎,15왼발목,16오른발목

# 3) 각도 계산(점 B에서의 ∠ABC)
def angle_deg(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = np.array([ax - bx, ay - by], dtype=np.float32)
    v2 = np.array([cx - bx, cy - by], dtype=np.float32)
    dot = (v1 * v2).sum()
    n1 = np.linalg.norm(v1) + 1e-6
    n2 = np.linalg.norm(v2) + 1e-6
    cos_t = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_t))

# 4) 캡처 시작
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # 5) 포즈 추론
    #   결과는 [Results]; 각 Results에 .keypoints (n,17,2)와 .boxes 포함
    res = model.predict(frame, imgsz=640, conf=0.35, iou=0.5, verbose=False)[0]

    # 6) 사람별로 스켈레톤/키포인트 그리기
    kps = res.keypoints  # ultralytics.engine.results.Keypoints
    if kps is not None:
        # xy: (n,17,2), conf: (n,17)
        xy = kps.xy.cpu().numpy() if hasattr(kps, "xy") else None
        kc = kps.conf.cpu().numpy() if getattr(kps, "conf", None) is not None else None

        # 바운딩박스/클래스도 그리려면 boxes 사용
        # boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []

        if xy is not None:
            for i, person in enumerate(xy):  # person shape: (17,2)
                # 키포인트 점
                for j, (px, py) in enumerate(person):
                    conf_j = kc[i, j] if kc is not None else 1.0
                    if conf_j < 0.25:  # 낮은 확률은 스킵
                        continue
                    cv2.circle(frame, (int(px), int(py)), 3, (0, 255, 0), -1)

                # 스켈레톤 선
                for (a, b) in SKELETON:
                    if a < len(person) and b < len(person):
                        pa = tuple(map(int, person[a]))
                        pb = tuple(map(int, person[b]))
                        cv2.line(frame, pa, pb, (0, 255, 255), 2)

                # (보너스) 왼/오른 팔꿈치 각도 계산 및 표기
                # 왼쪽: Shoulder(5)-Elbow(7)-Wrist(9), 오른쪽: 6-8-10
                pts = person
                def safe_get(idx): return tuple(map(float, pts[idx])) if idx < len(pts) else None

                L = [safe_get(5), safe_get(7), safe_get(9)]
                R = [safe_get(6), safe_get(8), safe_get(10)]
                if all(p is not None for p in L):
                    angL = angle_deg(L[0], L[1], L[2])
                    cv2.putText(frame, f"L-elbow: {angL:5.1f}°", (int(pts[7][0]) + 6, int(pts[7][1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                if all(p is not None for p in R):
                    angR = angle_deg(R[0], R[1], R[2])
                    cv2.putText(frame, f"R-elbow: {angR:5.1f}°", (int(pts[8][0]) + 6, int(pts[8][1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    # 7) FPS
    now = time.time()
    fps = 1.0 / max(now - prev, 1e-6)
    prev = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("YOLO Pose + OpenCV", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()