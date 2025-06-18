import os
import skvideo.io

def get_total_duration(folder_path):
    total_seconds = 0.0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.avi'):
                file_path = os.path.join(root, file)
                try:
                    # 读取视频元数据
                    metadata = skvideo.io.ffprobe(file_path)
                    video_info = metadata.get('video', {})
                    duration_str = video_info.get('@duration', None)
                    if not duration_str:
                        print(f"警告：文件 {file_path} 缺少时长信息，已跳过。")
                        continue
                    duration = float(duration_str)
                    total_seconds += duration
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错：{e}")
                    continue
    return total_seconds

if __name__ == "__main__":
    folder_path = input("请输入视频文件夹路径：")
    total_seconds = get_total_duration(folder_path)
    
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    print(f"\n总时长：{minutes} 分 {seconds:.2f} 秒")