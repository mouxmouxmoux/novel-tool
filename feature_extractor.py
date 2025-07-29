import hashlib
import numpy as np
from simhash import Simhash
from datasketch import MinHash, MinHashLSH
import re

class FeatureExtractor:
    def __init__(self, n_gram=3, num_perm=128):
        self.n_gram = n_gram
        self.num_perm = num_perm
    
    def get_ngrams(self, text, n):
        """获取文本的n-gram序列"""
        text = re.sub(r'\s+', '', text)  # 移除空白字符
        return [text[i:i+n] for i in range(len(text)-n+1)]
    
    def get_simhash(self, text):
        """计算文本的SimHash值"""
        features = self.get_ngrams(text, self.n_gram)
        return Simhash(features)
    
    def get_minhash(self, text):
        """计算文本的MinHash值"""
        m = MinHash(num_perm=self.num_perm)
        features = self.get_ngrams(text, self.n_gram)
        for feature in features:
            m.update(feature.encode('utf-8'))
        return m
    
    def calculate_similarity(self, hash1, hash2, method='simhash'):
        """计算两个哈希值的相似度"""
        if method == 'simhash':
            return hash1.distance(hash2) / 64  # SimHash默认64位
        elif method == 'minhash':
            return hash1.jaccard(hash2)
        return 0
    
    def extract_features(self, text):
        """提取文本特征"""
        return {
            'simhash': self.get_simhash(text),
            'minhash': self.get_minhash(text),
            'length': len(text)
        }