
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import tqdm
import time
# standard_colors = np.array([
#     [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],[133, 128, 177], [103, 189, 170], 
#     [214, 126, 44], [80, 91, 166],[193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
#     [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],[187, 86, 149], [8, 133, 161],
#     [243, 243, 242], [200, 200, 200],[160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52]
# ])
standard_colors = np.array([
    [103, 189, 170], [224, 163, 46],[8,133, 161],[52, 52, 52],
    [133, 128, 177],[157,188,64],[187,86,149],[85,85,85],
    [87, 108, 67],[94, 60, 108],[231,199,31],[122,122,121],
    [98, 122, 157], [193, 90, 99],[175,54,60],[160,160,160],
    [194, 150, 130],[80, 91, 166],[70,148,73],[200,200,200],
    [115, 82, 68],[214,126,44],[56,61,150],[243,243,242]
])
# # 检查颜色映射
# colors_normalized = standard_colors / 255.0
# # 创建4x6的网格
# fig, axes = plt.subplots(6, 4, figsize=(6, 8))
# fig.suptitle('4×6 colorcheck', fontsize=16, fontweight='bold')

# for i in range(6):
#     for j in range(4):
#         idx = i * 4 + j
#         color = colors_normalized[idx]
#         axes[i, j].imshow([[color]])
#         axes[i, j].set_title(f'({i},{j})\nRGB{standard_colors[idx]}', fontsize=8, pad=2)
#         axes[i, j].axis('off')
# plt.tight_layout()
# plt.show()

def calculate_correction_matrix(raw_values, target_values):
    raw_values = np.hstack([raw_values, np.ones((raw_values.shape[0], 1))])
    correction_matrix, _, _, _ = np.linalg.lstsq(raw_values, target_values, rcond=None)
    # print("correction_matrix\n",correction_matrix)
    return correction_matrix

def apply_correction(data, correction_matrix):
    original_shape = data.shape
    # print("original_shape",original_shape)
    # print("correction_matrix",correction_matrix.shape)
    data_flat = data.reshape(-1, 3)
    # print("data_flat",data_flat.shape)
    # 添加一列1以支持仿射变换
    data_flat = np.hstack([data_flat, np.ones((data_flat.shape[0], 1))])
    # print("data_flat",data_flat.shape)
    corrected_data_flat = np.dot(data_flat, correction_matrix)
    # 将结果重新调整为原始形状
    corrected_data = corrected_data_flat.reshape(original_shape)
    return corrected_data
def apply_correction_rgb(data, correction_matrix):
    original_shape = data.shape
    data_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data_flat = data_rgb.reshape(-1, 3)
    data_flat = np.hstack([data_flat, np.ones((data_flat.shape[0], 1))])
    corrected_data_flat = np.dot(data_flat, correction_matrix)
    corrected_data_flat = np.clip(corrected_data_flat, 0, 255)
    corrected_data_rgb = corrected_data_flat.reshape(original_shape).astype(np.uint8)
    corrected_data_bgr = cv2.cvtColor(corrected_data_rgb, cv2.COLOR_RGB2BGR)
    return corrected_data_bgr

def process_video_color_correction(df,input_video_path=None,output_video_path=None,max_frames=100):
    required_columns = ['frame']
    for i in range(1, 25):
        required_columns.extend([f'{i}_r', f'{i}_g', f'{i}_b'])
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误：CSV文件中缺少必要的列: {missing_columns}")
        return
    print("正在打开输入视频...")
    cap = cv2.VideoCapture(input_video_path)
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(total_frames, max_frames)
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"视频信息: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")

    print("开始颜色校正处理...")
    
    for frame_idx in tqdm.tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame_data = df[df['frame'] == frame_idx]
        if len(frame_data) > 0:
            current_colors = []
            for i in range(1, 25):
                r = frame_data[f'{i}_r'].values[0]
                g = frame_data[f'{i}_g'].values[0]
                b = frame_data[f'{i}_b'].values[0]
                current_colors.append([r, g, b])
            current_colors = np.array(current_colors, dtype=np.float32)
            correction_matrix = calculate_correction_matrix(current_colors, standard_colors)
            corrected_frame = apply_correction_rgb(frame, correction_matrix)
        else:
            corrected_frame = frame
        out.write(corrected_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"颜色校正完成！输出文件: {output_video_path}")

if __name__ == "__main__":
    max_frame = 16000
    t = time.time()
    csv_path = r"D:\codes\face_spo2\output\video_color_data.csv"
    input_video_path = r"D:\codes\face_spo2\output\video_ZIP_H264_1005161318\segment.mp4"
    output_video_path = r"D:\codes\face_spo2\output\video_ZIP_H264_1005161318\segment_corrected.mp4"
    if not os.path.exists(csv_path):
        print(f"错误：CSV文件不存在 - {csv_path}")
    if not os.path.exists(input_video_path):
        print(f"错误：输入视频文件不存在 - {input_video_path}")
    print("正在读取CSV文件...")
    df = pd.read_csv(csv_path)
    process_video_color_correction(df, input_video_path, output_video_path,max_frame)
    print(f"总耗时: {time.time()-t:.2f} 秒")