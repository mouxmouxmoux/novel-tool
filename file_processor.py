import os
import chardet
from zhconv import convert
import logging
from tqdm import tqdm

class FileProcessor:
    def __init__(self, source_dir, temp_dir):
        self.source_dir = source_dir
        self.temp_dir = temp_dir
        self.ensure_temp_dir()
        
    def ensure_temp_dir(self):
        """确保临时目录存在"""
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def detect_encoding(self, file_path):
        """检测文件编码"""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # 读取前10000字节用于检测
            result = chardet.detect(raw_data)
            return result['encoding']
    
    def convert_to_utf8(self, file_path, target_path):
        """将文件转换为UTF-8编码并统一简体"""
        try:
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
            
            # 繁体转简体
            content = convert(content, 'zh-cn')
            
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        except Exception as e:
            logging.error(f"转换文件 {file_path} 失败: {str(e)}")
            return False
    
    def scan_files(self):
        """扫描所有文本文件"""
        txt_files = []
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.lower().endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        return txt_files
    
    def process_files(self, file_list, state_manager):
        """批量处理文件"""
        processed_files = []
        
        for file_path in tqdm(file_list, desc="处理文件"):
            if state_manager.is_processed(file_path):
                continue
                
            # 生成临时文件路径
            rel_path = os.path.relpath(file_path, self.source_dir)
            temp_path = os.path.join(self.temp_dir, rel_path)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            
            # 转换编码
            if self.convert_to_utf8(file_path, temp_path):
                processed_files.append((file_path, temp_path))
                state_manager.mark_processed(file_path)
                
                # 定期保存状态
                if len(processed_files) % 100 == 0:
                    state_manager.save()
        
        state_manager.save()
        return processed_files