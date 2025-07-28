import os


def list_image_files(folder_path):
    """
    遍历指定文件夹并打印出所有图片文件的完整文件名。

    :param folder_path: 要扫描的文件夹路径。
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：找不到文件夹 '{folder_path}'")
        return

    print(f"--- 在文件夹 '{folder_path}' 中找到的图片文件 ---")

    # 定义常见的图片文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

    found_images = False
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否以定义的图片扩展名结尾（不区分大小写）
        if filename.lower().endswith(image_extensions):
            print(filename)
            found_images = True

    if not found_images:
        print("未找到任何图片文件。")


# --- 主程序入口 ---
if __name__ == "__main__":
    # ######################################################
    # ##  请在这里修改为您自己的图片文件夹路径          ##
    # ######################################################

    # Windows 路径示例: target_folder = r"C:\Users\YourName\Desktop\MyImages"
    # macOS/Linux 路径示例: target_folder = "/Users/YourName/Desktop/MyImages"
    target_folder = "../dataset/cls_train"

    list_image_files(target_folder)