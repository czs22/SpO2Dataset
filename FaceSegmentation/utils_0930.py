import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
from datetime import datetime

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV使用BGR格式
def get_polygon(rectangle_points, check_points,ratio=0.5):
    """
    根据两个参考点和四个检查点返回平移后的四个特征点
    
    参数:
        rectangle_points: 两个参考点的坐标列表 [(x151, y151), (x10, y10)]
        check_points: 四个检查点的坐标列表 [(x1, y1), (x2, y2), ...]
    
    返回:
        translated_points: 平移后的四个点的坐标列表 [(x1, y1), (x2, y2), ...]
    """
    if len(rectangle_points) != 2 or len(check_points) != 4:
        return None
    
    try:
        # 获取点151和点10的坐标
        x151, y151 = rectangle_points[0]
        x10, y10 = rectangle_points[1]
        
        # 计算平移向量（点151到点10方向的一半距离）
        dx = (x10 - x151)*ratio
        dy = (y10 - y151)*ratio
        
        # 计算平移后的四个点
        translated_points = []
        for (x, y) in check_points:
            new_x = int(x + dx)
            new_y = int(y + dy)
            translated_points.append((new_x, new_y))
        
        return translated_points
        
    except (IndexError, TypeError):
        return None

def perspective_transform(frame, points,output_size=128):
    """
    将四边形区域透视变换为64x64正方形
    
    参数:
        frame: 原始帧
        points: 四个点的坐标 [左上, 左下, 右上, 右下]
    
    返回:
        transformed: 变换后的64x64图像
        matrix: 透视变换矩阵
    """
    if len(points) != 4:
        return None, None
    reordered_points = [
            points[0],points[1],
            points[2], points[3]]
    points = reordered_points
    dst_points = np.array([
        [0, 0],                   # 左上
        [output_size-1, 0],       # 右上
        [output_size-1, output_size-1],  # 右下
        [0, output_size-1]        # 左下
    ], dtype=np.float32)
    src_points = np.array(points, dtype=np.float32)
    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    # 应用透视变换
    transformed = cv2.warpPerspective(frame, matrix, (output_size, output_size))
    return transformed, matrix

def create_segmented_video(frame, translated_points,output_size=128):
    """
    创建分割视频：只保留变换后四边形区域内的数据，其他部分为黑色
    并在右下角显示透视变换后的64x64图像
    
    参数:
        frame: 原始帧
        translated_points: 平移后的四个点坐标
    
    返回:
        segmented_frame: 处理后的帧
        transformed_img: 透视变换后的64x64图像
    """
    if translated_points is None or len(translated_points) != 4:
        print("Warning: Invalid translated points for segmentation.")
        segmented_frame = np.zeros_like(frame)
        transformed_img = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        return segmented_frame, transformed_img

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    # 在掩码上绘制四边形区域（白色）
    pts = np.array(translated_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)
    transformed_img, _ = perspective_transform(frame, translated_points,output_size)
    if transformed_img is not None:
        h, w = frame.shape[:2]
        x_start = w - output_size
        y_start = h - output_size
        if x_start >= 0 and y_start >= 0:
            segmented_frame[y_start:y_start+output_size, 
                           x_start:x_start+output_size] = transformed_img
    return segmented_frame, transformed_img
