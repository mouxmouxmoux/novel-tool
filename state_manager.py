import json
import os
from typing import Dict, List, Set, Any
from datetime import datetime
import numpy as np
from simhash import Simhash
from datasketch import MinHash
from redis_cache import RedisCache

class StateManager:
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.redis_cache = RedisCache()
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """加载状态"""
        # 首先尝试从Redis加载
        state = self.redis_cache.get_processing_state()
        if state:
            return state
            
        # 如果Redis中没有，则从文件加载
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    # 转换list为set
                    state['processed_files'] = set(state['processed_files'])
                    return state
            except Exception as e:
                print(f"加载状态文件失败: {e}")
        
        # 返回默认状态
        return {
            'processed_files': set(),
            'file_features': {},
            'clusters': {},
            'last_update': datetime.now().isoformat(),
            'current_phase': 'initialized'
        }
    
    def save(self):
        """保存状态"""
        # 保存到Redis
        self.redis_cache.save_processing_state(self.state)
        
        # 同时保存到文件作为备份
        # 转换set为list以便JSON序列化
        save_state = {
            'processed_files': list(self.state['processed_files']),
            'file_features': self.state['file_features'],
            'clusters': self.state['clusters'],
            'last_update': datetime.now().isoformat(),
            'current_phase': self.state['current_phase']
        }
        
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(save_state, f, ensure_ascii=False, indent=2)
    
    def is_processed(self, file_path: str) -> bool:
        """检查文件是否已处理"""
        # 首先检查Redis缓存
        if self.redis_cache.is_processed(file_path):
            return True
        return file_path in self.state['processed_files']
    
    def mark_processed(self, file_path: str):
        """标记文件为已处理"""
        if file_path not in self.state['processed_files']:
            self.state['processed_files'].add(file_path)
        
        # 同时标记到Redis缓存
        self.redis_cache.mark_processed(file_path)
    
    def save_file_features(self, file_path: str, features: Dict):
        """保存文件特征"""
        # 保存到Redis缓存
        self.redis_cache.save_file_features(file_path, features)
        
        # 将SimHash和MinHash转换为可序列化格式
        serializable_features = {
            'simhash': int(features['simhash'].value),
            'minhash': features['minhash'].hashvalues.tolist(),
            'length': features['length']
        }
        self.state['file_features'][file_path] = serializable_features
    
    def get_file_features(self) -> Dict[str, Dict]:
        """获取所有文件特征"""
        # 合并Redis缓存和本地状态中的特征
        features = self.state['file_features'].copy()
        
        # 注意：在实际使用中，可能需要从Redis获取更多特征
        # 这里简化处理，只使用本地状态中的特征
        
        # 重建SimHash和MinHash对象
        for file_path, feature in features.items():
            # 重建SimHash
            feature['simhash'] = Simhash(feature['simhash'], 64)
            
            # 重建MinHash
            m = MinHash(num_perm=128)
            m.hashvalues = np.array(feature['minhash'])
            feature['minhash'] = m
            
        return features
    
    def save_clusters(self, clusters: Dict[int, List[str]]):
        """保存聚类结果"""
        self.state['clusters'] = clusters
    
    def get_clusters(self) -> Dict[int, List[str]]:
        """获取聚类结果"""
        return self.state['clusters']
    
    def set_phase(self, phase: str):
        """设置当前处理阶段"""
        self.state['current_phase'] = phase
    
    def get_phase(self) -> str:
        """获取当前处理阶段"""
        return self.state['current_phase']
    
    def reset(self):
        """重置状态"""
        self.state = {
            'processed_files': set(),
            'file_features': {},
            'clusters': {},
            'last_update': datetime.now().isoformat(),
            'current_phase': 'initialized'
        }
        self.save()