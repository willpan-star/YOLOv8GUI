from pathlib import Path
import sys

# 获取当前文件的绝对路径
file_path = Path(__file__).resolve()

# 获取当前文件的父目录
root_path = file_path.parent

# 将根路径添加到 sys.path 列表（如果尚不存在）
if root_path not in sys.path:
    sys.path.append(str(root_path))

# 获取根目录相对于当前工作目录的相对路径
ROOT = root_path.relative_to(Path.cwd())


# 检测资源类型
SOURCES_LIST = ["Image", "Video", "Webcam", "Folder", "Dataset"]


# 检测模型配置
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
YOLOv8n_best = DETECTION_MODEL_DIR / "yolov8n_best.pt"
YOLOv8s_best = DETECTION_MODEL_DIR / "yolov8s_best.pt"


DETECTION_MODEL_LIST = [
    "yolov8n_best.pt",
    "yolov8s_best.pt",
    ]
