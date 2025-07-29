import json
import os
from typing import Dict, List, Set, Any
from datetime import datetime
import numpy as np
from simhash import Simhash
from datasketch import MinHash

class StateManager:
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """加载状态文件"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载状态文件失败: {e}")
        
        # 返回默认状态
        return {
            'processed_files': [],
            'file_features': {},
            'clusters': {},
            'last_update': datetime.now().isoformat(),
            'current_phase': 'initialized'
        }
    
    def save(self):
        """保存状态到文件"""
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
        return file_path in self.state['processed_files']
    
    def mark_processed(self, file_path: str):
        """标记文件为已处理"""
        if file_path not in self.state['processed_files']:
            self.state['processed_files'].append(file_path)
    
    def save_file_features(self, file_path: str, features: Dict):
        """保存文件特征"""
        # 将SimHash和MinHash转换为可序列化格式
        serializable_features = {
            'simhash': int(features['simhash'].value),
            'minhash': features['minhash'].hashvalues.tolist(),
            'length': features['length']
        }
        self.state['file_features'][file_path] = serializable_features
    
    def get_file_features(self) -> Dict[str, Dict]:
        """获取所有文件特征"""
        return self.state['file_features']
    
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
            'processed_files': [],
            'file_features': {},
            'clusters': {},
            'last_update': datetime.now().isoformat(),
            'current_phase': 'initialized'
        }
        self.save()