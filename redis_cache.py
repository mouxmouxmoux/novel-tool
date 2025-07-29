import redis
import json
import numpy as np
from typing import Dict, Any, Optional
from simhash import Simhash
from datasketch import MinHash
import logging
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, USE_REDIS

class RedisCache:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB):
        """
        初始化Redis缓存连接
        
        Args:
            host: Redis服务器主机名
            port: Redis服务器端口
            db: Redis数据库编号
        """
        if not USE_REDIS:
            self.enabled = False
            self.cache = {}
            return
            
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
            # 测试连接
            self.redis_client.ping()
            self.enabled = True
        except Exception as e:
            logging.warning(f"无法连接到Redis服务器: {e}. 将使用内存缓存.")
            self.enabled = False
            self.cache = {}
    
    def save_file_features(self, file_path: str, features: Dict[str, Any]) -> bool:
        """
        保存文件特征到Redis缓存
        
        Args:
            file_path: 文件路径
            features: 特征字典
            
        Returns:
            bool: 保存是否成功
        """
        if not self.enabled:
            self.cache[file_path] = features
            return True
            
        try:
            # 将特征转换为可序列化格式
            serializable_features = {
                'simhash': int(features['simhash'].value),
                'minhash': features['minhash'].hashvalues.tolist(),
                'length': features['length']
            }
            
            # 保存到Redis
            key = f"file_features:{file_path}"
            self.redis_client.set(key, json.dumps(serializable_features))
            return True
        except Exception as e:
            logging.error(f"保存文件特征到Redis失败: {e}")
            return False
    
    def get_file_features(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        从Redis缓存获取文件特征
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 特征字典，如果不存在则返回None
        """
        if not self.enabled:
            return self.cache.get(file_path)
            
        try:
            key = f"file_features:{file_path}"
            data = self.redis_client.get(key)
            if data:
                features = json.loads(data)
                # 重建SimHash对象
                features['simhash'] = Simhash(features['simhash'], 64)
                
                # 重建MinHash对象
                m = MinHash(num_perm=128)
                m.hashvalues = np.array(features['minhash'])
                features['minhash'] = m
                
                return features
        except Exception as e:
            logging.error(f"从Redis获取文件特征失败: {e}")
        
        return None
    
    def save_processing_state(self, state: Dict[str, Any]) -> bool:
        """
        保存处理状态到Redis缓存
        
        Args:
            state: 状态字典
            
        Returns:
            bool: 保存是否成功
        """
        if not self.enabled:
            self.cache['processing_state'] = state
            return True
            
        try:
            # 转换set为list以便JSON序列化
            save_state = {
                'processed_files': list(state['processed_files']),
                'clusters': state['clusters'],
                'last_update': state['last_update'],
                'current_phase': state['current_phase']
            }
            
            self.redis_client.set('processing_state', json.dumps(save_state))
            return True
        except Exception as e:
            logging.error(f"保存处理状态到Redis失败: {e}")
            return False
    
    def get_processing_state(self) -> Optional[Dict[str, Any]]:
        """
        从Redis缓存获取处理状态
        
        Returns:
            Dict[str, Any]: 状态字典，如果不存在则返回None
        """
        if not self.enabled:
            return self.cache.get('processing_state')
            
        try:
            data = self.redis_client.get('processing_state')
            if data:
                state = json.loads(data)
                # 转换list为set
                state['processed_files'] = set(state['processed_files'])
                return state
        except Exception as e:
            logging.error(f"从Redis获取处理状态失败: {e}")
        
        return None
    
    def is_processed(self, file_path: str) -> bool:
        """
        检查文件是否已处理
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否已处理
        """
        state = self.get_processing_state()
        if state and 'processed_files' in state:
            return file_path in state['processed_files']
        return False
    
    def mark_processed(self, file_path: str) -> bool:
        """
        标记文件为已处理
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 标记是否成功
        """
        state = self.get_processing_state()
        if not state:
            state = {
                'processed_files': set(),
                'clusters': {},
                'last_update': '',
                'current_phase': 'initialized'
            }
        
        if file_path not in state['processed_files']:
            state['processed_files'].add(file_path)
            return self.save_processing_state(state)
        return True