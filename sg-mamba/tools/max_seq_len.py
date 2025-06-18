import numpy as np
import os

def main():
    dir_path = "./data/mma52/feats/224x16x4x1_vit-g"
    max_shape0 = 0
    file_count = 0
    
    print(f"开始扫描目录: {dir_path}")
    
    for filename in os.listdir(dir_path):
        if not filename.endswith(".npy"):
            continue
            
        filepath = os.path.join(dir_path, filename)
        file_count += 1
        
        try:
            # 使用mmap_mode可以更快加载大文件且节省内存
            data = np.load(filepath, mmap_mode="r")
            current_shape = data.shape
            if len(current_shape) < 1:
                print(f"警告: {filename} 是零维数组，已跳过")
                continue
                
            current_shape0 = current_shape[0]
            if current_shape0 > max_shape0:
                max_shape0 = current_shape0
                print(f"更新最大值: {max_shape0} (来自 {filename})")
                
        except Exception as e:
            print(f"加载文件 {filename} 时出错: {str(e)}")
            continue

    print("\n统计完成！")
    print(f"已扫描 .npy 文件总数: {file_count}")
    if file_count > 0:
        print(f"所有文件中 shape[0] 的最大值为: {max_shape0}")
    else:
        print("警告: 目录中没有找到任何.npy文件")

if __name__ == "__main__":
    main()