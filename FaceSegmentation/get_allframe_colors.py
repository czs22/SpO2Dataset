import cv2
import numpy as np
import pandas as pd

def load_perspective_data(csv_path):
    """加载透视变换数据"""
    df = pd.read_csv(csv_path)
    points_data = []
    for _, row in df.iterrows():
        points_data.append({
            'id': row['id'],
            'x': row['x'],
            'y': row['y'],
            'r': row['r'],
            'g': row['g'],
            'b': row['b']})
    return points_data

def extract_color_data(video_path, points_data, max_frames=None):
    """提取视频中每个帧的颜色数据"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    print(f"视频总帧数: {total_frames}")
    # 准备存储所有帧的颜色数据
    all_frame_data = []
    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_data = {'frame': frame_count + 1}
        # 提取每个点的颜色
        for point in points_data:
            x, y, point_id = int(point['x']), int(point['y']), int(point['id'])
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                b, g, r = frame[y, x]
            else:
                r, g, b = 0, 0, 0
            frame_data[f'{point_id}_r'] = r
            frame_data[f'{point_id}_g'] = g
            frame_data[f'{point_id}_b'] = b
        all_frame_data.append(frame_data)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"处理进度: {frame_count}/{total_frames}")
    cap.release()
    return all_frame_data

def save_color_data(all_frame_data, output_csv):
    """保存颜色数据到CSV文件"""
    df = pd.DataFrame(all_frame_data)
    # 确保列的顺序：frame, 1_r, 1_g, 1_b, 2_r, 2_g, 2_b, ...
    columns = ['frame']
    for i in range(1, 25):
        columns.extend([f'{i}_r', f'{i}_g', f'{i}_b'])
    df = df.reindex(columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"颜色数据已保存到: {output_csv}")
    print(f"总帧数: {len(df)}")
    print(f"数据形状: {df.shape}")

def main():
    csv_path = r"D:\codes\face_spo2\colorcheck_frame1.csv"
    video_path = r"D:\codes\face_spo2\data\070507\v01\video_ZIP_H264.avi"
    output_csv = r"D:\codes\face_spo2\video_color_data.csv"
    MAX_FRAME = 100
    print("加载透视变换数据...")
    points_data = load_perspective_data(csv_path)
    print(f"成功加载 {len(points_data)} 个采样点")
    
    print("提取视频颜色数据...")
    all_frame_data = extract_color_data(video_path, points_data, MAX_FRAME)
    if all_frame_data:
        print("保存颜色数据...")
        save_color_data(all_frame_data, output_csv)
        # 显示前几行数据预览
        df_preview = pd.read_csv(output_csv)
        print("\n数据预览 (前5行):")
        print(df_preview.head())
        print("\n数据列名:")
        print(df_preview.columns.tolist())
    else:
        print("未能提取颜色数据")
if __name__ == "__main__":
    main()