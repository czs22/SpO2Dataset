
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
from datetime import datetime
from utils_0930 import *
import pandas as pd
rectangle_landmarks = [151,10]
rect_colorlist = ["#0011FF", "#8800FF"]
check_landmarks = [109,338,336,107]
check_colorlist =  ['#FF8400', '#FFA500', '#AAAA00', '#00FF00']
MAX_FRAMES = 100

rect_color = [hex_to_bgr(color) for color in rect_colorlist]
check_color = [hex_to_bgr(color) for color in check_colorlist]
OUTPUT_SIZE = 64

landmark_displacements = []

def draw_displacement_chart(frame, displacements, max_frames=MAX_FRAMES, max_dist=20, colors=check_color):
    """
    在frame左下角绘制位移曲线图
    displacements: List[List[float]]，每帧每个点的位移
    """
    chart_height = 100
    chart_width = 200
    margin = 20
    n_points = len(check_landmarks)
    h, w = frame.shape[:2]
    x0 = margin
    y0 = h - chart_height - margin
    chart = np.ones((chart_height, chart_width, 3), dtype=np.uint8) * 255

    n_frames = len(displacements)
    if n_frames < 2:
        frame[y0:y0+chart_height, x0:x0+chart_width] = chart
        return frame
    
    # 横坐标: 时间 0 ~ max_frames
    # 纵坐标: 位移 0 ~ max_disp
    for i in range(n_points):
        pts = []
        for t, disp in enumerate(displacements):
            x = int(t / max_frames * (chart_width-1))
            y = int(chart_height - 1 - min(disp[i], max_dist) / max_dist * (chart_height-1))
            pts.append((x, y))
        for j in range(1, len(pts)):
            cv2.line(chart, pts[j-1], pts[j], colors[i], 1)
    
    # 坐标轴
    cv2.rectangle(chart, (0,0), (chart_width-1, chart_height-1), (0,0,0), 1)
    # y轴刻度
    for yval in range(0, max_dist+1, 10):
        y = int(chart_height - 1 - yval / max_dist * (chart_height-1))
        cv2.line(chart, (0, y), (5, y), (0,0,0), 1)
        cv2.putText(chart, str(yval), (7, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
    # x轴刻度
    for xval in range(0, max_frames+1, max(1, max_frames//4)):
        x = int(xval / max_frames * (chart_width-1))
        cv2.line(chart, (x, chart_height-1), (x, chart_height-6), (0,0,0), 1)
        cv2.putText(chart, str(xval), (x, chart_height-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
    
    # 图表叠加到frame左下角
    frame[y0:y0+chart_height, x0:x0+chart_width] = chart
    return frame

def calculate_displacements(current_points, previous_points, frame_count,csv_path):
    """
    计算当前帧与上一帧的位移，并输出详细信息
    """
    displacements = []
    if previous_points is None or frame_count == 0:# 第一帧，位移为0
        displacements = [0.0 for _ in current_points]
        mean_displacement = 0.0
    else:
        for i, (curr_pt, prev_pt) in enumerate(zip(current_points, previous_points)):
            displacement = np.linalg.norm(np.array(curr_pt) - np.array(prev_pt))
            displacements.append(displacement)
    mean_displacement = np.mean(displacements) if displacements else 0.0
    # 输出总览信息
    row = {'frame': frame_count, 'mean_dist': mean_displacement}
    for i, (idx, (x, y)) in enumerate(zip(check_landmarks, current_points)):
        row[f"{idx}_x"] = x
        row[f"{idx}_y"] = y
        # row[f"{idx}_dist"] = displacements[i]
    # 检查文件是否存在，决定是否写入表
    file_exists = os.path.isfile(csv_path)
    df_row = pd.DataFrame([row])
    if file_exists:
        df_row.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode='w', header=True, index=False)
    return displacements,mean_displacement

def create_segmented_and_dist(frame, check_points=None, translated_points=None, check_color=None, 
                             landmark_displacements=[], previous_points=None, frame_count=0,mean_displacement=None):
    """
    创建分割视频：只保留变换后四边形区域内的数据，其他部分为黑色
    并在右下角显示透视变换后的64x64图像
    并在左下角绘制landmark位移曲线
    """
    if translated_points is None or len(translated_points) != 4:
        print("Warning: Invalid translated points for segmentation.")
        segmented_frame = np.zeros_like(frame)
        transformed_img = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.uint8)
        return segmented_frame, transformed_img

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(translated_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)
    transformed_img, _ = perspective_transform(frame, translated_points, OUTPUT_SIZE)
    if transformed_img is not None:
        h, w = frame.shape[:2]
        x_start = w - OUTPUT_SIZE
        y_start = h - OUTPUT_SIZE
        if x_start >= 0 and y_start >= 0:
            segmented_frame[y_start:y_start+OUTPUT_SIZE,x_start:x_start+OUTPUT_SIZE] = transformed_img
    if check_points is not None:# 收集所有帧的位移
        dist_list = [item['dist'] for item in landmark_displacements]
        segmented_frame = draw_displacement_chart(segmented_frame, dist_list, 
                max_frames=MAX_FRAMES, max_dist=20,colors=check_color)
    return segmented_frame,transformed_img

def draw_landmarks_and_dist(frame, check_points=None, translated_points=None, check_color=None, 
                           landmark_displacements=[], previous_points=None, frame_count=0,mean_displacement=None):
    """
    在帧上绘制四边形和位移信息
    """
    if check_points is None or translated_points is None or len(check_points) != 4:
        return frame
    for i, (x, y) in enumerate(check_points):# 绘制landmark点
        color = check_color[i]
        cv2.circle(frame, (x, y), 3, color, -1)
        label = f"{check_landmarks[i]}"
        cv2.putText(frame, label, (x+8, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    # 绘制平移后的四边形（黄色边框）
    pts_translated = np.array(translated_points, np.int32)
    pts_translated = pts_translated.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts_translated], True, (0, 255, 255), 1)  # 黄色边框
    h, w = frame.shape[:2]
    text_y = 50
    if landmark_displacements and 'mean_dist' in landmark_displacements[-1]:
        mean_dist = landmark_displacements[-1]['mean_dist']
    else:
        mean_dist = 0.0
    if mean_dist < 3:
        mean_dist_str = "基本静止"
        text_color = (0, 255, 0)
    elif mean_dist < 7:
        mean_dist_str = "轻微移动"
        text_color = (0, 255, 255)
    elif mean_dist < 10:
        mean_dist_str = "中度移动"
        text_color = (0, 165, 255)
    else:
        mean_dist_str = "剧烈移动"
        text_color = (0, 0, 255)
    cv2.putText(frame,f'Frame {frame_count},mean:{mean_dist} {mean_dist_str}',                       
                (w - 300, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    dist_list = [item['dist'] for item in landmark_displacements]
    frame = draw_displacement_chart(frame, dist_list, 
                max_frames=MAX_FRAMES, max_dist=50,colors=check_color)
    return frame

def process_video_with_landmarks(video_path, output_folder, model_path):
    output_video_path = os.path.join(output_folder, "landmarks.mp4")
    output_masked_path = os.path.join(output_folder, "masked.mp4")
    output_segmented_path = os.path.join(output_folder, "segment.mp4")
    output_csv_path = os.path.join(output_folder, "landmarks.csv")

    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(total_frames, MAX_FRAMES)
    print(f"视频信息: {frame_width}x{frame_height}, FPS: {fps:.2f}, 总帧数: {total_frames}")
    print(f"监测的landmarks: {check_landmarks}")
    print("=" * 80)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    output_masked = cv2.VideoWriter(output_masked_path, fourcc, fps, (frame_width, frame_height))
    output_segmented = cv2.VideoWriter(output_segmented_path, fourcc, fps, (OUTPUT_SIZE, OUTPUT_SIZE))
    # 初始化MediaPipe FaceLandmarker
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,num_faces=1)
    frame_count = 0
    landmark_displacements = []
    previous_points = None
    with FaceLandmarker.create_from_options(options) as landmarker:
        while frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                print("视频处理完成")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
            face_landmarks_list = results.face_landmarks
            if face_landmarks_list:
                for face_landmarks in face_landmarks_list:
                    h, w, _ = frame.shape
                    rectangle_points = []
                    check_points = []
                    for landmark_idx in rectangle_landmarks:
                        if landmark_idx < len(face_landmarks):
                            landmark = face_landmarks[landmark_idx]
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            rectangle_points.append((x, y))
                    for landmark_idx in check_landmarks:
                        if landmark_idx < len(face_landmarks):
                            landmark = face_landmarks[landmark_idx]
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            check_points.append((x, y))

                    if len(rectangle_points) == 2 and len(check_points) == 4:
                        translated_points = get_polygon(rectangle_points, check_points, 0.6)
                        displacements,mean_dist = calculate_displacements(check_points, previous_points, frame_count,output_csv_path)
                        landmark_displacements.append({
                            'points': check_points, 
                            'dist': displacements,
                            'mean_dist': mean_dist,
                            'frame_count': frame_count})
                        annotated_frame = draw_landmarks_and_dist(
                            frame.copy(), check_points, translated_points, 
                            check_color, landmark_displacements, previous_points, frame_count)
                        segmented_frame,transformed_img = create_segmented_and_dist(
                            frame.copy(), check_points, translated_points, 
                            check_color, landmark_displacements, previous_points, frame_count,mean_dist)
                        previous_points = check_points
                    else:
                        print(f"Warning: Not enough landmarks detected in frame {frame_count}")
                        annotated_frame = frame.copy()
                        segmented_frame = np.zeros_like(frame)
                        transformed_img = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.uint8)
            else:
                print(f"Warning: No face landmarks detected in frame {frame_count}")
                annotated_frame = frame.copy()
                segmented_frame = np.zeros_like(frame)
                # 如果没有检测到landmarks，位移设为0
                if frame_count == 0:
                    displacements = [0.0 for _ in check_landmarks]
                else:
                    displacements = [0.0 for _ in check_landmarks]
                landmark_displacements.append({
                    'points': previous_points if previous_points else [(0,0) for _ in check_landmarks],
                    'dist': displacements,
                    'mean_dist': mean_dist,
                    'frame_count': frame_count})
            out_video.write(annotated_frame)
            output_masked.write(segmented_frame)
            output_segmented.write(transformed_img)
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%)")
    
    cap.release()
    out_video.release()
    output_masked.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 80)
    print("位移统计总结:")
    total_displacements = [item['dist'] for item in landmark_displacements if item['dist']]
    if total_displacements:
        avg_displacements = np.mean(total_displacements, axis=0)
        max_displacements = np.max(total_displacements, axis=0)
        for i, landmark_idx in enumerate(check_landmarks):
            print(f"Landmark {landmark_idx}: 平均位移 {avg_displacements[i]:.2f}px, "
                  f"最大位移 {max_displacements[i]:.2f}px")
    print(f"处理完成！输出视频保存至: {output_video_path}")
    print(f"总共处理了 {frame_count} 帧")

if __name__ == "__main__":
    video_path = r"D:\codes\face_spo2\data\070507\v01\video_ZIP_H264.avi"
    video_name = os.path.basename(video_path).split('.')[0]
    output_dir = r"D:\codes\face_spo2\output"
    model_path = r"D:\codes\face_spo2\files\face_landmarker_v2_with_blendshapes.task"

    t0 = time.time()
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    output_folder = f"D:/codes/face_spo2/output/{video_name}_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    process_video_with_landmarks(video_path, output_folder, model_path)
    print(f"总耗时: {time.time() - t0:.2f} 秒")