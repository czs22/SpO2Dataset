import cv2
import numpy as np
import pandas as pd
import os
# 全局变量

Width = 100
Height = int(Width*1.5)
def mouse_callback(event, x, y, flags, param):
    global points, click_count
    
    if event == cv2.EVENT_LBUTTONDOWN and click_count < 4:
        points.append((x, y))
        click_count += 1
        print(f"点击了第 {click_count} 个点: ({x}, {y})")
        # 在图像上绘制点和序号
        cv2.circle(frame_copy, (x, y), 1, (0, 255, 0), -1)
        cv2.putText(frame_copy, str(click_count), (x+3, y+3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # 连接点形成四边形
        if len(points) > 1:
            for i in range(len(points)-1):
                cv2.line(frame_copy, points[i], points[i+1], (255, 0, 0), 1)
        cv2.imshow("Select 4 Points", frame_copy)

def perspective_transform(image, src_points, width=Width, height=Height):
    """透视变换到指定大小的矩形"""
    # 目标点坐标
    dst_points = np.array([[0, 0], [width-1, 0], 
                          [width-1, height-1], [0, height-1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.array(src_points, dtype=np.float32), dst_points)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix

def inverse_transform_points(sampling_points, perspective_matrix, width=Width, height=Height):
    """将采样点反变换回原始图像坐标"""
    # 创建逆变换矩阵
    inverse_matrix = cv2.getPerspectiveTransform(
        np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32),
        np.array(points, dtype=np.float32))
    original_points = []
    for (x, y) in sampling_points:
        point = np.array([x, y, 1], dtype=np.float32)
        # 应用逆变换
        transformed = inverse_matrix @ point
        # 转换回笛卡尔坐标
        x_orig = int(transformed[0] / transformed[2])
        y_orig = int(transformed[1] / transformed[2])
        original_points.append((x_orig, y_orig))
    
    return original_points

def sample_colors(warped_image, width=40, height=60):
    """在变换后的图像上采样24个位点的RGB值"""
    x_coords = [int(width*5/40),int(width*15/40),int(width*25/40),int(width*35/40)]
    y_coords = [int(height*5/60),int(height*15/60),int(height*25/60),int(height*35/60),int(height*45/60), int(height*55/60)]
    sampling_points = []
    colors = []
    for y in y_coords:
        for x in x_coords:
            if y < warped_image.shape[0] and x < warped_image.shape[1]:
                # 获取RGB值 (OpenCV使用BGR格式)
                b, g, r = warped_image[y, x]
                sampling_points.append((x, y))
                colors.append((r, g, b))
    return sampling_points, colors

def main(video_path,output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    global frame_copy, points, click_count
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # cap.release()
    frame_copy = frame.copy()
    # 显示图像并设置鼠标回调
    cv2.imshow("Select 4 Points", frame_copy)
    cv2.setMouseCallback("Select 4 Points", mouse_callback)
    print("请依次点击四个点来定义四边形（顺时针或逆时针）")
    print("按ESC键退出，按任意键继续")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            cv2.destroyAllWindows()
            return
        elif click_count >= 4:
            break
    cv2.destroyWindow("Select 4 Points")
    warped, matrix = perspective_transform(frame, points,Width, Height)
    sampling_points, colors = sample_colors(warped,Width, Height)
    original_points = inverse_transform_points(sampling_points, matrix, Width, Height)
    
    data = []
    for i, ((x_orig, y_orig), (r, g, b)) in enumerate(zip(original_points, colors)):
        data.append({
            'id': i+1,
            'x': x_orig,
            'y': y_orig,
            'r': r,
            'g': g,
            'b': b})
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_folder, 'colorcheck_frame1.csv')
    df.to_csv(csv_path, index=False)
    # 在变换后的图像上标记采样点
    target_frames = [1, 7500, 15000]
    marked_frames = []
    # 处理每个目标帧
    for i, frame_num in enumerate(target_frames):
        # 设置视频到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # 帧号从0开始
        ret, current_frame = cap.read()
        if not ret:
            print(f"警告: 无法读取第 {frame_num} 帧，视频可能没有这么多帧")
            continue
        # 在当前帧上标记原始坐标点
        frame_with_points = current_frame.copy()
        for j, (x, y) in enumerate(original_points):
            cv2.circle(frame_with_points, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(frame_with_points, str(j+1), (x+3, y-3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # 保存标记后的图片
        output_image_path = os.path.join(output_folder, f'colorcheck_frame_{frame_num}.png')
        cv2.imwrite(output_image_path, frame_with_points)
        marked_frames.append((frame_num, frame_with_points))
        print(f"已保存第 {frame_num} 帧的标记图片")
    cap.release()

    for frame_num, marked_frame in marked_frames:
        window_name = f"Frame {frame_num} with Sampling Points"
        cv2.imshow(window_name, marked_frame)
        # 调整窗口位置，避免重叠
        if frame_num == 1:
            cv2.moveWindow(window_name, 100, 100)
        elif frame_num == 7500:
            cv2.moveWindow(window_name, 600, 100)
        elif frame_num == 15000:
            cv2.moveWindow(window_name, 1100, 100)
    warped_display = warped.copy()
    for i, (x, y) in enumerate(sampling_points):
        cv2.circle(warped_display, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(warped_display, str(i+1), (x+2, y-2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    # 新增：在原图上标记原始坐标点
    original_with_points = frame.copy()
    for i, (x, y) in enumerate(original_points):
        cv2.circle(original_with_points, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(original_with_points, str(i+1), (x+3, y-3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imshow("Original Frame with Sampling Points", original_with_points)
    # 把图片保存到output_folder
    cv2.imshow("Original Frame with Points", frame_copy)
    cv2.imshow("Perspective Transformed", warped_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points, warped, sampling_points, colors,original_points

if __name__ == "__main__":
    output_base = 'D:/codes/face_spo2/output/colorcheck_marks'
    file_list = 'E:/experiment/data/file_list.csv'
    input_base = 'E:/experiment/data'
    df_files = pd.read_csv(file_list)
    for file_path in df_files['file_path']:
        points = []
        click_count = 0
        video_folder = os.path.join(input_base, file_path)
        video_path = os.path.join(video_folder, "video_ZIP_H264.avi")
        print(video_path)
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
        output_folder = os.path.join(output_base, file_path)
        main(video_path, output_folder)