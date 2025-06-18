import json
from pathlib import Path

def convert_json_format(input_json, output_json):
    """
    转换标注文件格式
    :param input_json: 原处理脚本生成的annotations.json路径
    :param output_json: 输出文件路径
    """
    # 加载原始数据
    with open(input_json, 'r') as f:
        original_data = json.load(f)
    
    converted_data = {}
    
    # 遍历所有视频片段
    for video_name, clips in original_data.items():
        for clip_info in clips:
            # 获取裁剪视频文件名（不带后缀）
            clip_key = Path(clip_info["video_path"]).stem
            
            # 构建新的数据结构
            converted_data[clip_key] = {
                "subset": clip_info.get("subset", "train"),
                "duration": clip_info["duration"],
                "fps": clip_info["fps"],
                "original_window": clip_info["original_window"],
                "annotations": clip_info["annotations"]
            }
    
    # 保存转换结果
    with open(output_json, 'w') as f:
        json.dump(converted_data, f, indent=2)
    print(f"转换完成！共处理 {len(converted_data)} 个视频片段")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='JSON格式转换工具')
    parser.add_argument('--input', required=True, help='原始annotations.json路径')
    parser.add_argument('--output', required=True, help='输出文件路径')
    
    args = parser.parse_args()
    
    # 示例用法：
    # python convert_script.py --input ./output/annotations.json --output ./output/converted.json
    convert_json_format(args.input, args.output)