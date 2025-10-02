额头关键区域提取
segment_and_speed_1002.py
加载视频，保存：
  特征点位置landmarks.csv
  64*64视频segment.mp4
  （可选）原始视频+关键点标注landmarks.mp4
  （可选）视频mask,保留：框选区域、右下角显示透视变换后图像，左下显示运动速度masked.mp4
  
colorcheck提取
mark_colorcheck.py
加载视频，显示第一帧
手动点选四角，自动识别四角位置
保存成colorcheck_frame1.csv

get_allframe.py
加载colorcheck_frame1.csv和视频
提取色卡位点的颜色，保存到video_color_data.csv
