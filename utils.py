import io
import zipfile
import imageio
import yaml
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import logging

logging.disable(logging.CRITICAL)  # 禁用所有级别的日志


IMG_Type = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
VIDEO_Type = ['.mp4', '.mov', '.avi', '.flv']

basePath = os.getcwd()
resultBasePath = basePath + "/results"
if not os.path.exists(resultBasePath):
    os.mkdir(resultBasePath)


# 绘制PR图、F1-Score图等
def drawChart(model, source_yaml):
    # --- 修改yaml当中的path --- #
    yaml_path = None
    file_contents = source_yaml.getvalue() # 读取上传的文件
    yaml_data = yaml.safe_load(file_contents) # 解析YAML内容
    if 'path' in yaml_data:
        yaml_path = f"{yaml_data['path']}/{source_yaml.name}"
    # --- 修改yaml当中的path --- #
    if yaml_path:
        result = model.val(data=yaml_path)
        resutl_pic_path = basePath + "/" +str(result.save_dir)
        return resutl_pic_path
    else:
        return list()

# 用于检测图片，显示检测结果
def detect_image(model, conf, image_name, uploaded_image, right_image_st, detection_result_st):
    res = model.predict(uploaded_image,
                        conf=conf)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]

    # 使用 imageio.imwrite 保存图片
    imageio.imwrite(f"{resultBasePath}/{image_name}", res_plotted)

    right_image_st.image(res_plotted,
             caption="Detected Image",
             use_column_width=True)
    try:
        with detection_result_st.expander("检测结果"):
            for box in boxes:
                st.write(box.xywh)
    except Exception as ex:
        right_image_st.write("暂无图片上传!")
        with right_image_st:
            st.write(ex)

# 检测视频
def detect_video(model, source_video, conf, video_name, right_image_st):
    try:
        # stream读取文件
        tfile = tempfile.NamedTemporaryFile()
        tfile.write(source_video.read())
        vid_cap = cv2.VideoCapture(
            tfile.name)

        # 获取视频宽度、高度和帧率
        width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vid_cap.get(cv2.CAP_PROP_FPS)

        # 定义VideoWriter用于保存视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编解码器
        video_name = f"{resultBasePath}/{video_name}"
        out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        st_frame = right_image_st
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                # 在这里处理每一帧
                processed_image = _display_detected_frames(conf, model, st_frame, image)
                # 将处理后的帧写入视频
                out.write(processed_image)
            else:
                vid_cap.release()
                out.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {e}")


# 用于文件夹保存所有图片，压缩成一个压缩包
def create_zip_file(paths):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for path in paths:
            # 添加图片到ZIP文件，os.path.basename获取文件名
            zip_file.write(path, os.path.basename(path))
    zip_buffer.seek(0)
    return zip_buffer

def _display_detected_frames(conf, model, st_frame, image):
    """
    将yolov8模型检测的目标展示在视频框架中
    一个视频有很多秒 ==> 每一秒有很多帧 ==> 每一帧就是一张图片
    """
    res_out = model.predict(image, conf=conf)

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    return res_out[0].plot()

@st.cache_resource
def load_model(model_path):
    """
    从给定模型路径中加载YOLO目标i检测模型
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):

    source_img = st.sidebar.file_uploader(
        label="选择一张图片...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    # --- 创建占位符 --- #
    col1, col2 = st.columns(2)
    left_image_placeholder = col1.empty()
    right_image = col2.empty()
    detection_result = col2.empty()
    # --- 创建占位符 --- #

    with left_image_placeholder:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                detect_image(model, conf, image_name=source_img.name, uploaded_image=uploaded_image,
                             right_image_st=right_image, detection_result_st=detection_result)
            down_button = st.empty()
            with down_button:
                image = Image.open(f'{resultBasePath}/{source_img.name}')
                # 将图像转换为二进制数据
                buf = io.BytesIO()
                image.save(buf, format='JPEG' if source_img.name.endswith(".jpg") else "PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label=f"Save Result",
                    data=byte_im,
                    file_name=f"{source_img.name}",
                    mime="image/jpeg" if source_img.name.endswith(".jpg") else "image/png"
                )

def infer_uploaded_video(conf, model):
    """
    执行接口加载视频 Execute inference for uploaded video
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video...",
        type=("asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm")
    )
    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                right_image_st = st.empty()
                detect_video(model, source_video, conf, video_name=source_video.name, right_image_st=right_image_st)
            down_button = st.empty()
            with down_button:
                # Open the video file in binary mode
                with open(f'{resultBasePath}/{source_video.name}', 'rb') as f:
                    video_data = f.read()

                st.download_button(
                    label=f"Save Result",
                    data=video_data,
                    file_name=f"{source_video.name}",
                    mime="video/mp4"  # Assuming the video is in mp4 format
                )


def infer_uploaded_webcam(conf, model):
    """
    执行摄像头 Execute inference for webcam.
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")

def infer_uploaded_folder(conf, model):
    """
        Execute inference for uploaded image
        """
    # 设置侧边栏上传
    uploaded_files = st.sidebar.file_uploader("Choose a folder", accept_multiple_files=True,
                                              type=IMG_Type + VIDEO_Type)
    if uploaded_files:
        # 执行按钮状态
        # --- 创建占位符 --- #
        col1, col2 = st.columns(2)
        left_image_placeholder = col1.empty()
        right_image= col2.empty()
        detection_result = col2.empty()
        # --- 创建占位符 --- #

        first_uploaded_file = uploaded_files[0]
        first_fielname = first_uploaded_file.name

        source_img = None
        uploaded_image = None
        source_video = None

        # 判断是否为图片, 如果是图片就显示第一张图片
        if any(first_fielname.lower().endswith(ext) for ext in IMG_Type):
            source_img = first_uploaded_file
            with left_image_placeholder:
                if source_img:
                    uploaded_image = Image.open(source_img)
                    # adding the uploaded image to the page with caption
                    st.image(
                        image=source_img,
                        caption="Uploaded Image",
                        use_column_width=True
                    )
        elif any(first_fielname.lower().endswith(ext) for ext in VIDEO_Type):
            source_video = first_uploaded_file
            with left_image_placeholder:
                if source_video:
                    st.video(source_video)

        if st.button("Execution", key="Execution"):  # 点击了Execution
            with st.spinner("Running..."):
                # 检测第一个 图片 / 视频
                if any(first_fielname.lower().endswith(ext) for ext in IMG_Type):
                    detect_image(model, conf, image_name=source_img.name, uploaded_image=uploaded_image,
                                 right_image_st=right_image, detection_result_st=detection_result)
                elif any(first_fielname.lower().endswith(ext) for ext in VIDEO_Type):
                    detection_result.empty()
                    detect_video(model, source_video, conf, video_name=source_video.name, right_image_st=right_image)

                # 依次检测后面的视频和图片
                for uploaded_file in uploaded_files[1:]:
                    filename = uploaded_file.name
                    # 判断是否为图片
                    if any(filename.lower().endswith(ext) for ext in IMG_Type):
                        source_img = uploaded_file
                        with left_image_placeholder:
                            if source_img:
                                uploaded_image = Image.open(source_img)
                                # adding the uploaded image to the page with caption
                                st.image(
                                    image=source_img,
                                    caption="Uploaded Image",
                                    use_column_width=True
                                )
                        if source_img:
                            # 最后面的两个参数是占位符
                            detect_image(model, conf, image_name=source_img.name, uploaded_image=uploaded_image,
                                         right_image_st=right_image, detection_result_st=detection_result)
                    # 判断是否为视频
                    elif any(filename.lower().endswith(ext) for ext in VIDEO_Type):
                        source_video = uploaded_file
                        with left_image_placeholder:
                            if source_video:
                                st.video(source_video)
                        if source_video:
                            detection_result.empty()
                            detect_video(model, source_video, conf, video_name=source_video.name,
                                         right_image_st=right_image)
            down_button = st.empty()
            with down_button:
                image_names = [image_upload.name for image_upload in uploaded_files]
                image_paths = [f"{resultBasePath}/{image_name}" for image_name in image_names]
                # 将图像压缩成压缩包
                result_zip_buffer = create_zip_file(image_paths)
                st.download_button(
                    label=f"Save Result",
                    data=result_zip_buffer,
                    file_name=f"results.zip",
                    mime="application/zip"
                )

def infer_uploaded_yaml(model):
    source_yaml = st.sidebar.file_uploader(
        label="Choose an dataset yaml...",
        type=('yaml')
    )

    if source_yaml:
        result_pic_path = drawChart(model, source_yaml)
        pics_name = os.listdir(result_pic_path)
        col1, col2 = st.columns(2)  # 第一行
        col3, col4 = st.columns(2)  # 第二行
        results_name = []
        for pic_name in pics_name:
            if "F1_curve" in pic_name:
                with col1:
                    st.image(f"{result_pic_path}/{pic_name}", caption="F1-Score Curve", use_column_width=True)
                results_name.append(pic_name)
            elif "PR_curve" in pic_name:
                with col2:
                    st.image(f"{result_pic_path}/{pic_name}", caption="PR Curve", use_column_width=True)
                results_name.append(pic_name)
            elif "P_curve" in pic_name:
                with col3:
                    st.image(f"{result_pic_path}/{pic_name}", caption="Precision Curve", use_column_width=True)
                results_name.append(pic_name)
            elif "R_curve" in pic_name:
                with col4:
                    st.image(f"{result_pic_path}/{pic_name}", caption="Recall Curve", use_column_width=True)
                results_name.append(pic_name)
        results_name = [f"{result_pic_path}/{pic_name}" for pic_name in results_name]
        down_button = st.empty()
        with down_button:
            # 将图像压缩成压缩包
            result_zip_buffer = create_zip_file(results_name)
            st.download_button(
                label=f"Save Result",
                data=result_zip_buffer,
                file_name=f"resultCharts.zip",
                mime="application/zip"
            )
        source_yaml = None
