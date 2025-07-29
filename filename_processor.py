import re
import os
from zhconv import convert

class FilenameProcessor:
    def __init__(self):
        # 定义需要移除的模式（如第几章、作者名等）
        self.remove_patterns = [
            r'第[\d\-]+章',
            r'第[\d\-]+节',
            r'[\d\-]+章',
            r'[\d\-]+节',
            r'\d+-?\d*',
            r'卷\d+',
            r'部\d+',
            r'（.*?）',  # 括号内容
            r'\(.*?\)',   # 英文括号内容
            r'【.*?】',   # 方括号内容
            r'作者[:：]?\s*[^，,]+',
            r'by\s+.*',
            r'更新时间[:：]?\s*\d+[-年]\d+[-月]\d+[日]?',
            r'\d+年\d+月\d+日',
            r'正文卷',
            r'VIP卷',
            r'最新章节',
            r'全文完',
        ]
        
        # 编译正则表达式
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.remove_patterns]
    
    def clean_filename(self, filename):
        """
        清理文件名，移除章节号、作者信息等非核心内容
        """
        # 获取不带扩展名的文件名
        basename = os.path.splitext(filename)[0]
        
        # 繁体转简体
        basename = convert(basename, 'zh-cn')
        
        # 移除不需要的模式
        for pattern in self.compiled_patterns:
            basename = pattern.sub('', basename)
        
        # 移除多余的空白字符和特殊符号
        basename = re.sub(r'[_\-—\.\s]+', ' ', basename)
        basename = re.sub(r'^\s+|\s+$', '', basename)  # 去除首尾空格
        basename = re.sub(r'\s+', ' ', basename)  # 合并多个空格
        
        # 如果清理后名称太短，则使用原始文件名
        if len(basename) < 2:
            return convert(os.path.splitext(filename)[0], 'zh-cn')
        
        return basename
    
    def calculate_filename_similarity(self, filename1, filename2):
        """
        计算两个文件名的相似度
        返回值: 0-1之间的浮点数，1表示完全相同
        """
        # 清理文件名
        clean_name1 = self.clean_filename(filename1)
        clean_name2 = self.clean_filename(filename2)
        
        # 如果清理后完全相同
        if clean_name1 == clean_name2:
            return 1.0
        
        # 计算字符级别的相似度（使用简单的公共子序列方法）
        # 将字符串转换为字符集合
        set1 = set(clean_name1)
        set2 = set(clean_name2)
        
        # 计算交集和并集
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        # Jaccard相似度
        if len(union) == 0:
            return 1.0 if clean_name1 == clean_name2 else 0.0
            
        jaccard_similarity = len(intersection) / len(union)
        
        # 如果一个名称包含另一个名称
        if clean_name1 in clean_name2 or clean_name2 in clean_name1:
            return max(jaccard_similarity, 0.8)  # 至少返回0.8的相似度
            
        return jaccard_similarity
    
    def is_similar_filename(self, filename1, filename2, threshold=0.6):
        """
        判断两个文件名是否相似
        """
        similarity = self.calculate_filename_similarity(filename1, filename2)
        return similarity >= threshold