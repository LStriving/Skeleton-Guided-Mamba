import json
from collections import defaultdict

def add_alltime_annotation(input_path, output_path):
    # 读取原始标注数据
    with open(input_path) as f:
        data = json.load(f)
    
    for clip_id in data:
        clip = data[clip_id]
        annotations = clip["annotations"]
        fps = clip["fps"]
        
        # 缓存不同吞咽阶段的动作组
        swallow_groups = defaultdict(list)
        
        # 根据时间重叠聚类分组
        for ann in sorted(annotations, key=lambda x: x["segment"][0]):
            matched = False
            # 检查是否属于已有吞咽组
            for group_id in list(swallow_groups.keys()):
                group = swallow_groups[group_id]
                last_end = max(a["segment"][1] for a in group)
                if ann["segment"][0] <= last_end:
                    swallow_groups[group_id].append(ann)
                    matched = True
                    break
            
            # 未匹配则创建新组
            if not matched:
                swallow_groups[len(swallow_groups)].append(ann)
        
        # 生成AllTime标注
        valid_swallows = []
        for group_id, group_anns in swallow_groups.items():
            # 验证是否包含完整的吞咽阶段（1-7）
            label_ids = {a["label_id"] for a in group_anns}
            if label_ids == set(range(1, 8)):
                # 计算时间范围
                min_start = min(a["segment"][0] for a in group_anns)
                max_end = max(a["segment"][1] for a in group_anns)
                
                # 转换帧数
                start_frame = round(min_start * fps)
                end_frame = round(max_end * fps)
                
                valid_swallows.append({
                    "label": "AllTime",
                    "segment": [min_start, max_end],
                    "segment(frames)": [start_frame, end_frame],
                    "label_id": 0
                })
        
        # 添加有效吞咽标注
        clip["annotations"].extend(valid_swallows)
    
    # 保存处理后的标注
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='add all time')
    parser.add_argument('--input', required=True, help='原始annotations.json路径')
    parser.add_argument('--output', required=True, help='输出文件路径')
    
    args = parser.parse_args()
    add_alltime_annotation(
        args.input,
        args.output
    )