
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"GPU 개수: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 버전: {torch.version.cuda}")