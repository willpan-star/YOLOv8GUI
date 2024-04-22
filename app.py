from pathlib import Path

import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam, infer_uploaded_folder, \
    infer_uploaded_yaml
#  设置页面布局
st.set_page_config(
    page_title="安全帽检测界面",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# 主页标题设置
# st.title("YOLOv8 交互式界面")

# 侧边栏
st.sidebar.header("检测模型配置")

# 模型选择
task_type = st.sidebar.selectbox(
    "任务选择Task",
    ["Detection"]
)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "模型选择",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("目前仅实现检测功能")

confidence = float(st.sidebar.slider(
    "置信度调整", 30, 100, 43)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("请在侧边栏选择使用的模型")

# 加载训练好的检测模型
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"无法加载模型，请核实好模型确切的路径！: {model_path}")

# 图像/视频 选择
st.sidebar.header("图像/视频 配置")
source_selectbox = st.sidebar.selectbox(
    "选择来源",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # 图片
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # 视频
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # 网络摄像头
    infer_uploaded_webcam(confidence, model)
elif source_selectbox == config.SOURCES_LIST[3]: # 文件夹
    infer_uploaded_folder(confidence, model)
elif source_selectbox == config.SOURCES_LIST[4]: #数据集
    infer_uploaded_yaml(model)
else:
    st.error("目前仅供检测图片以及视频")