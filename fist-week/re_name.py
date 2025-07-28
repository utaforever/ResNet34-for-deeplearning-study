import os


def rename_images(folder_path):
    """
    批量重命名指定文件夹内的图片文件。

    :param folder_path: 包含图片的目标文件夹路径。
    """
    # --- 安全检查 ---
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return

    # 1. 获取文件夹内所有文件名
    try:
        filenames = os.listdir(folder_path)
    except Exception as e:
        print(f"错误：无法读取文件夹内容。 {e}")
        return

    # 2. 筛选出图片文件（可以根据需要添加更多格式）
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    image_files = [f for f in filenames if f.lower().endswith(image_extensions)]

    # 3. 对文件名进行排序，确保重命名顺序可预测
    image_files.sort()

    # 4. 循环遍历并重命名
    count = 1
    for old_name in image_files:
        # 分离文件名和扩展名，以便保留原始格式
        file_extension = os.path.splitext(old_name)[1]

        # 构建新的文件名，例如：h_01.jpg, h_02.png 等
        # f'{count:02d}' 会将数字格式化为两位，不足则前面补0 (例如 1 -> 01)
        new_name = f'h_{count:02d}{file_extension}'

        # 获取完整的旧路径和新路径
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        # 执行重命名
        try:
            os.rename(old_path, new_path)
            print(f"成功: '{old_name}' -> '{new_name}'")
            count += 1
        except Exception as e:
            print(f"失败: 重命名 '{old_name}' 时出错。 {e}")

    print(f"\n处理完成！总共重命名了 {count - 1} 个图片文件。")


# --- 主程序入口 ---
if __name__ == "__main__":
    # ######################################################
    # ##  请在这里修改为您自己的图片文件夹路径          ##
    # ######################################################

    # Windows 路径示例: folder_path = r"C:\Users\YourName\Desktop\MyImages"
    # macOS/Linux 路径示例: folder_path = "/Users/YourName/Desktop/MyImages"
    folder_path = "../dataset/HuTao"

    rename_images(folder_path)