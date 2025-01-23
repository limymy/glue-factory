import os
import shutil
import tarfile
from pathlib import Path
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

class RevisitopDownloader:
    def __init__(self, data_dir="data", image_dir="", max_workers=5):
        self.conf = {
            "data_dir": data_dir,
            "image_dir": image_dir,
            "max_workers": max_workers  # 最大并发线程数
        }
        self.DATA_PATH = Path(os.getcwd())  # 设置当前工作目录为基础路径

    def _download_and_extract(self, n, url_base, image_dir):
        """下载并解压单个文件"""
        tar_name = f"revisitop1m.{n + 1}.tar.gz"
        tar_path = image_dir / tar_name
        # 下载文件
        torch.hub.download_url_to_file(url_base + "jpg/" + tar_name, tar_path)
        # 解压文件
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=image_dir)
        return tar_name

    def download_revisitop1m(self):
        # 数据存储路径
        data_dir = self.DATA_PATH / self.conf["data_dir"]
        tmp_dir = data_dir.parent / "revisitop1m_tmp"
        
        # 如果临时目录存在，清理上次失败的下载
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        
        # 创建临时图像存储目录
        image_dir = tmp_dir / self.conf["image_dir"]
        image_dir.mkdir(exist_ok=True, parents=True)
        
        # 数据文件和 URL 基础路径
        num_files = 100
        url_base = "http://ptak.felk.cvut.cz/revisitop/revisitop1m/"
        list_name = "revisitop1m.txt"
        
        # 下载文件列表
        print("Downloading file list...")
        torch.hub.download_url_to_file(url_base + list_name, tmp_dir / list_name)
        
        # 使用线程池进行并发下载
        print("Downloading and extracting files...")
        with ThreadPoolExecutor(max_workers=self.conf["max_workers"]) as executor:
            futures = [
                executor.submit(self._download_and_extract, n, url_base, image_dir)
                for n in range(num_files)
            ]
            for future in tqdm(as_completed(futures), total=num_files, desc="Processing"):
                try:
                    tar_name = future.result()
                    print(f"{tar_name} processed successfully.")
                except Exception as e:
                    print(f"Error processing a file: {e}")
        
        # 移动临时目录到最终数据目录
        shutil.move(tmp_dir, data_dir)
        print(f"Download complete. Data is saved at {data_dir}")

# 示例调用
if __name__ == "__main__":
    downloader = RevisitopDownloader(max_workers=2)  # 设置最大并发下载数
    downloader.download_revisitop1m()
