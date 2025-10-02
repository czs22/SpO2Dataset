import cv2
import numpy as np
import pandas as pd
import os
# 全局变量
points = []
click_count = 0
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
    global frame_copy, points, click_count
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return
    cap.release()
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
    df.to_csv("colorcheck_frame1.csv", index=False)
    # 在变换后的图像上标记采样点
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
    output_image_path = os.path.join(output_folder, 'frame1_colorcheck.png')
    cv2.imwrite(output_image_path, original_with_points)

    cv2.imshow("Original Frame with Points", frame_copy)
    cv2.imshow("Perspective Transformed", warped_display)
    print("\n24个采样点的RGB值：")
    print("序号\tX\tY\tR\tG\tB")
    for i, ((x, y), (r, g, b)) in enumerate(zip(original_points, colors)):
        print(f"{i+1:2d}\t{x}\t{y}\t{r:3d}\t{g:3d}\t{b:3d}")
    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points, warped, sampling_points, colors,original_points

if __name__ == "__main__":
    video_path = r"D:\codes\face_spo2\data\070507\v01\video_ZIP_H264.avi"
    output_folder = r"D:\codes\face_spo2\output"
    main(video_path,output_folder)