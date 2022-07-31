
import os

from isort import file

def remove(path):
    os.remove(path)

if __name__ == "__main__":
    root_dir = input("输入清楚文件目录:").strip()
    filenames = os.listdir(root_dir)
    file_paths = [os.path.join(root_dir,name) for name in filenames]
    for f in file_paths:
        remove(f)