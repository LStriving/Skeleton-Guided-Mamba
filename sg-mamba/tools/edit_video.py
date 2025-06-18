# pip install moviepy
from moviepy.editor import *
from moviepy.video.tools.drawing import color_gradient
import os

# 设置ImageMagick参数
os.environ["MAGICK_TMPDIR"] = "/tmp/moviepy"
os.makedirs("/tmp/moviepy", exist_ok=True)

def safe_add_intro(input_video, output_video, intro_duration=2):
    # 显式指定ImageMagick路径
    TextClip._IMAGEMAGICK_EXE = "/usr/bin/convert" 
    
    # 创建带权限的文字剪辑
    txt = TextClip("Intelligent VFSS Video Analysis Platform",
                   fontsize=40, color='white',
                   font='Arial-Bold',  # 使用明确字体名称
                   method='label',     # 避免管道操作
                   size=(1280, 720))   # 显式设置分辨率
    # 加载原始视频
    video = VideoFileClip(input_video)
    
    # 创建黑色背景（匹配视频尺寸）
    bg_clip = ColorClip(video.size, color=(0, 0, 0), duration=intro_duration)
    
    # 创建文字剪辑（默认使用Arial字体，可修改）
    txt_clip = TextClip("Intelligent VFSS Video Analysis Platform", 
                        fontsize=40, 
                        color='white', 
                        font='Arial',
                        size=video.size).set_duration(intro_duration)
    
    # 合成背景和文字
    intro = CompositeVideoClip([bg_clip, txt_clip.set_position('center')])
    
    # 为原始视频添加渐入效果（1秒淡入）
    faded_video = video.crossfadein(1.0)
    
    # 拼接开场和主视频（注意保持音频同步）
    final = concatenate_videoclips([intro, faded_video], 
                                 transition=None, 
                                 padding=-0.5)  # 补偿交叉淡入的帧重叠
    
    # 输出最终视频（保留原始音频）
    final.write_videofile(output_video, 
                         codec='libx264', 
                         audio_codec='aac', 
                         fps=video.fps)
    
video_path = 'assets/system.mp4'
output_path = video_path.replace(".mp4", "_edit.mp4")
safe_add_intro(video_path, output_path)
print(f'Video edited saved to {output_path}')