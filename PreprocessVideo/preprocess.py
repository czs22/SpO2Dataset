import os
import random
import subprocess
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import shutil

def log_message(message, log_file=None):
    """打印并保存日志"""
    print(message)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} - {message}\n")

def pic2video(stats_csv, data_root, fps=30, log_path=None, output_root=None): 
    # 用dataroot和stats_csv中的文件名拼成视频所在的路径，按照fps=30拼接图片为视频
    """
    对统计文件中的每个 file_path 进行预处理:
    1. 检查是否存在 video_RAW_RGBA.avi
    2. 如果不存在，使用 ffmpeg 将 pictures_ZIP_RAW_RGB 转换为 video_RAW_RGBA.avi
    3. （已确认相同）调用 validate_video_frames 进行抽帧验证
    """
    df = pd.read_csv(stats_csv)

    for _, row in df.iterrows():
        folder_path = os.path.join(data_root, row["file_path"].replace("/", os.sep))
        video_path = os.path.join(folder_path, "video_RAW_RGBA.avi")
        img_dir = os.path.join(folder_path, "pictures_ZIP_RAW_RGB")
        output_path = os.path.join(output_root, row["file_path"].replace("/", os.sep), "video_RAW_RGBA.avi") 
        log_message(f"\n📂 正在处理: {folder_path}", log_path)

        # 检查 video 是否存在，如果存在就不生成了
        if os.path.isfile(video_path):
            log_message("✅ 已存在 video_RAW_RGBA.avi，跳过生成，直接复制", log_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(video_path, output_path)
            log_message(f"✅ 已复制到: {output_path}", log_path)

        else:
            log_message("🚀 未找到 video_RAW_RGBA.avi，开始生成...", log_path)

            cmd = [ # 已测试，命令能够保持帧与图像完全对应
                "ffmpeg",
                "-framerate", str(fps),
                "-start_number", "0",
                "-i", os.path.join(img_dir, "%08d.png"),
                "-c:v", "rawvideo",
                "-pix_fmt", "rgba",
                output_path
            ]

            try:
                subprocess.run(cmd, check=True)
                log_message(f"🎉 成功生成: {output_path}", log_path)
            except subprocess.CalledProcessError as e:
                log_message(f"❌ ffmpeg 生成失败: {e}", log_path)
                continue

def interpolate(stats_csv, data_root, output_root): # 生理数据插值到帧时间戳
    """
    遍历 stats_csv 中的文件夹，检查是否有 HR/BVP/SpO2/RR 数据
    若存在，对应数据插值到 frames_timestamp.csv 的时间戳上
    输出 {datatype}_interp.csv
    """
    datatypes = ["HR", "BVP", "SpO2", "RR"]
    df = pd.read_csv(stats_csv)

    for _, row in df.iterrows():
        folder_path = os.path.join(data_root, row["file_path"].replace("/", os.sep))
        if not os.path.isdir(folder_path):
            continue

        frame_ts_file = os.path.join(folder_path, "frames_timestamp.csv")
        if not os.path.isfile(frame_ts_file):
            continue

        # 加载 frame timestamp
        try:
            df_frames = pd.read_csv(frame_ts_file)
            frame_ts = df_frames["timestamp"].values.astype(float)
        except Exception as e:
            print(f"⚠️ 无法读取 {frame_ts_file}: {e}")
            continue

        for datatype in datatypes:
            data_file = os.path.join(folder_path, f"{datatype}.csv")
            output_path = os.path.join(output_root, row["file_path"].replace("/", os.sep)) 
            os.makedirs(output_path, exist_ok=True)
            if not os.path.isfile(data_file):
                continue
                
            try:
                df_data = pd.read_csv(data_file)
                data_ts = df_data["timestamp"].values.astype(float)
                data_val = df_data[datatype.lower()].values.astype(float)

                # 插值
                data_interp = np.interp(frame_ts, data_ts, data_val)

                # 保存
                out_file = os.path.join(output_path, f"{datatype}.csv")
                pd.DataFrame({
                    f"{datatype.lower()}": data_interp,
                    "timestamp": frame_ts
                }).to_csv(out_file, index=False)


                print(f"✅ 已保存 {out_file}")

            except Exception as e:
                print(f"❌ 插值失败 {data_file}: {e}")

def copy_zip_videos(stats_csv, data_root, output_root, log_path=None):
    """
    遍历 stats_csv 中的 file_path 列，查找每个文件夹下的 video_ZIP_H264.avi，
    并复制到 output_root 的对应路径下。
    """
    df = pd.read_csv(stats_csv)

    for _, row in df.iterrows():
        folder_path = os.path.join(data_root, row["file_path"].replace("/", os.sep))
        input_path = os.path.join(folder_path, "video_ZIP_H264.avi")
        output_path = os.path.join(output_root, row["file_path"].replace("/", os.sep), "video_ZIP_H264.avi")

        if os.path.isfile(input_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(input_path, output_path)
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"📂 复制 {input_path} -> {output_path}\n")
            print(f"✅ 复制成功: {input_path} -> {output_path}")
        else:
            print(f"⚠️ 未找到 {input_path}")


def main():
    stats_csv = r"/root/jjt/SpO2Dataset/PreprocessVideo/test.csv"
    data_root = r"/root/datasets/rSpO2/data"
    output_root = r"/root/jjt/dataset_out"
    log_path = r"/root/jjt/pre.log"

    # 生成 RGBA 视频到 output_root
    print("\n=== Step 1: pic2video ===")
    pic2video(stats_csv, data_root, output_root=output_root, fps=30, log_path=log_path)

    # 生理数据插值并另存到 output_root
    print("\n=== Step 2: interpolate ===")
    interpolate(stats_csv, data_root, output_root=output_root)

    # 复制 ZIP 视频到 output_root
    print("\n=== Step 3: copy zip videos ===")
    copy_zip_videos(stats_csv, data_root, output_root, log_path=log_path)


if __name__ == "__main__":
    main()
