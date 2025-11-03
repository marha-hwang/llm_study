from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')
results = model.track("https://www.youtube.com/watch?v=MPyvBYaCoLc", show=True)


