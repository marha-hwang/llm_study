import cv2, time
from ultralytics import YOLO

#  
model = YOLO("yolo11n.pt")   # 이미 설치되어 있으면 자동 다운로드/캐시
names = model.names          # {클래스ID: 이름} 딕셔너리

# if CUDA
# model.to("cuda")

# ② 캡처 시작
cap = cv2.VideoCapture(0)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # ③ 한 프레임 추론
    #    결과는 [Results] 리스트, 여기선 단일 프레임이므로 results[0]만 사용
    results = model.predict(
        frame, imgsz=640, conf=0.35, iou=0.5, verbose=False
    )
    r = results[0]

    # ④ 박스/레이블 그리기 (xyxy 좌표)
    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            label  = f"{names.get(cls_id, cls_id)} {conf:.2f}"

            # 사각형 + 라벨
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

    # FPS 표시
    now = time.time()
    fps = 1.0 / max(now - prev, 1e-6)
    prev = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("YOLO + OpenCV", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()