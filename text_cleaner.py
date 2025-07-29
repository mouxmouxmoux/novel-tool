import re
from typing import List, Tuple

class TextCleaner:
    def __init__(self):
        # 预编译正则表达式
        self.ad_patterns = [
            r'广告\s*[:：]?\s*.*?(?=\n\n|\n\s*\n|$)',  # 广告标记
            r'www\.[^\s]+',                          # 网址
            r'http[s]?://[^\s]+',                    # URL
            r'本书由.*?整理制作',                     # 制作信息
            r'更多精彩.*?请访问.*?',                  # 推广信息
            r'【.*?广告.*?】',                        # 方括号广告
            r'^\s*第.*?章.*?$',                      # 章节标题（可选）
            r'请支持正版阅读',                       # 盗版提示
            r'支持正版文学',                         # 正版提示
            r'请到[\w\.]+下载正版',                  # 下载提示
            r'内容版权归作者所有',                   # 版权声明
            r'内容均来自互联网',                     # 来源声明
            r'内容来自于互联网',                     # 来源声明
            r'版权归原作者所有',                   # 版权声明
            r'版权归原网站所有',                   # 版权声明
            r'请勿转载',                           # 转载声明
            r'谢绝转载',                           # 转载声明
            r'拒绝转载',                           # 转载声明
            r'(第?\s*\d+\s*[章回节]\s*){2,}',        # 重复章节标题
        ]
        
        self.comment_patterns = [
            r'读者评论\s*[:：]?\s*.*?(?=\n\n|\n\s*\n|$)',
            r'网友留言\s*[:：]?\s*.*?(?=\n\n|\n\s*\n|$)',
            r'【.*?评论.*?】',
            r'【.*?留言.*?】',
            r'评论区.*?(?=\n\n|\n\s*\n|$)',
            r'热门评论.*?(?=\n\n|\n\s*\n|$)',
            r'网友[∶:：].*?[:：].*?(?=\n\n|\n\s*\n|$)',  # 网友对话
        ]
        
        self.author_note_patterns = [
            r'作者的话\s*[:：]?\s*.*?(?=\n\n|\n\s*\n|$)',
            r'作者留言\s*[:：]?\s*.*?(?=\n\n|\n\s*\n|$)',
            r'【.*?作者.*?】',
            r'作者[∶:：].*?(?=\n\n|\n\s*\n|$)',
            r'作者简介\s*[:：]?.*?(?=\n\n|\n\s*\n|$)',
            r'作者信息\s*[:：]?.*?(?=\n\n|\n\s*\n|$)',
        ]
        
        self.garbage_patterns = [
            r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.,，。！？?、；：:;""''()（）【】\[\]{}《》\-\+\*/%=<>~`@#$%^&\|\\]+',  # 非中英文数字标点
            r'\s{2,}',                               # 多个空格
            r'\n{3,}',                               # 多个换行
            r'[◆●○■□△▽▲▼★☆♀♂※☀☁♠♥♦♣♬♪♫♭♯✓✔✕✖✗✘✚✜✢✣✤✥✦✧✨✩✪✫✬✭✮✯✰✱✲✳✴✵✶✷✸✹✺✻✼✽✾✿❀❁❂❃❄❅❆❇❈❉❊❋]+',  # 特殊符号
        ]
        
        # 编译所有正则表达式
        self.compiled_patterns = []
        for patterns in [self.ad_patterns, self.comment_patterns, 
                        self.author_note_patterns, self.garbage_patterns]:
            compiled = [re.compile(p, re.MULTILINE | re.DOTALL | re.IGNORECASE) for p in patterns]
            self.compiled_patterns.append(compiled)
    
    def clean_text(self, text: str) -> Tuple[str, float]:
        """清洗文本，返回清洗后的文本和正文比例"""
        original_length = len(text)
        
        # 移除广告
        for pattern in self.compiled_patterns[0]:
            text = pattern.sub('', text)
        
        # 移除评论
        for pattern in self.compiled_patterns[1]:
            text = pattern.sub('', text)
        
        # 移除作者留言
        for pattern in self.compiled_patterns[2]:
            text = pattern.sub('', text)
        
        # 移除垃圾字符
        for pattern in self.compiled_patterns[3]:
            text = pattern.sub('', text)
        
        # 计算正文比例
        cleaned_length = len(text)
        content_ratio = cleaned_length / original_length if original_length > 0 else 0
        
        # 规范化空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text, content_ratio
    
    def is_valid_content(self, text: str, min_ratio: float = 0.5) -> bool:
        """判断文本是否有效（正文内容比例足够）"""
        cleaned_text, ratio = self.clean_text(text)
        return ratio >= min_ratio and len(cleaned_text) >= 100