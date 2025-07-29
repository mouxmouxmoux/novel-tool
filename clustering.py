import numpy as np
from collections import defaultdict
from datasketch import MinHashLSH
from typing import Dict, List, Tuple, Set
import os
from filename_processor import FilenameProcessor

class TextClusterer:
    def __init__(self, similarity_threshold=0.85, min_duplicate_length=500):
        self.similarity_threshold = similarity_threshold
        self.min_duplicate_length = min_duplicate_length
        self.clusters = defaultdict(list)
        self.lsh = MinHashLSH(threshold=1-similarity_threshold, num_perm=128)
        self.filename_processor = FilenameProcessor()
    
    def cluster_by_filename(self, file_features: Dict[str, Dict]) -> Dict[int, List[str]]:
        """
        基于文件名相似性进行初步聚类
        """
        # 按文件长度排序，优先处理长文件
        sorted_files = sorted(file_features.items(), 
                            key=lambda x: x[1]['length'], reverse=True)
        
        clusters = defaultdict(list)
        processed_files = set()
        cluster_id = 0
        
        for file_path1, features1 in sorted_files:
            if file_path1 in processed_files:
                continue
                
            # 创建新集群
            current_cluster = [file_path1]
            processed_files.add(file_path1)
            filename1 = os.path.basename(file_path1)
            
            # 查找相似文件名的文件
            for file_path2, features2 in sorted_files:
                if file_path2 in processed_files:
                    continue
                    
                filename2 = os.path.basename(file_path2)
                # 如果文件名相似，加入同一集群
                if self.filename_processor.is_similar_filename(filename1, filename2, threshold=0.6):
                    current_cluster.append(file_path2)
                    processed_files.add(file_path2)
            
            # 只有当集群包含多个文件时才添加
            if len(current_cluster) > 1:
                clusters[cluster_id] = current_cluster
                cluster_id += 1
                
        return clusters
    
    def cluster_by_simhash(self, file_features: Dict[str, Dict]):
        """基于SimHash进行聚类"""
        # 按文件长度排序，优先处理长文件
        sorted_files = sorted(file_features.items(), 
                            key=lambda x: x[1]['length'], reverse=True)
        
        cluster_id = 0
        for file_path, features in sorted_files:
            if features['length'] < self.min_duplicate_length:
                continue
                
            simhash = features['simhash']
            assigned = False
            
            # 检查是否属于现有集群
            for cid, files in self.clusters.items():
                # 取集群中第一个文件作为代表
                rep_file = files[0]
                rep_features = file_features[rep_file]
                rep_simhash = rep_features['simhash']
                
                # 计算相似度
                distance = simhash.distance(rep_simhash)
                similarity = 1 - distance / 64
                
                if similarity >= self.similarity_threshold:
                    self.clusters[cid].append(file_path)
                    assigned = True
                    break
            
            # 如果不属于任何现有集群，创建新集群
            if not assigned:
                self.clusters[cluster_id] = [file_path]
                cluster_id += 1
        
        return self.clusters
    
    def cluster_by_minhash_lsh(self, file_features: Dict[str, Dict]):
        """基于MinHash LSH进行聚类"""
        # 重建LSH索引
        self.lsh = MinHashLSH(threshold=1-self.similarity_threshold, num_perm=128)
        
        # 插入所有MinHash
        for file_path, features in file_features.items():
            if features['length'] >= self.min_duplicate_length:
                self.lsh.insert(file_path, features['minhash'])
        
        # 查询相似文件
        clusters = defaultdict(list)
        processed = set()
        
        for file_path, features in file_features.items():
            if file_path in processed or features['length'] < self.min_duplicate_length:
                continue
                
            # 查询相似文件
            result = self.lsh.query(features['minhash'])
            
            # 创建集群
            cluster_id = len(clusters)
            for similar_file in result:
                if similar_file not in processed:
                    clusters[cluster_id].append(similar_file)
                    processed.add(similar_file)
        
        return clusters
    
    def merge_small_clusters(self, clusters: Dict[int, List[str]], 
                           file_features: Dict[str, Dict], min_cluster_size=2):
        """合并小集群"""
        # 找出小集群
        small_clusters = {cid: files for cid, files in clusters.items() 
                         if len(files) < min_cluster_size}
        
        # 尝试将小集群合并到大集群
        for small_cid, small_files in small_clusters.items():
            for small_file in small_files:
                small_features = file_features[small_file]
                small_simhash = small_features['simhash']
                
                best_match = None
                best_similarity = 0
                
                # 寻找最相似的大集群
                for big_cid, big_files in clusters.items():
                    if len(big_files) >= min_cluster_size:
                        rep_file = big_files[0]
                        rep_features = file_features[rep_file]
                        rep_simhash = rep_features['simhash']
                        
                        similarity = 1 - small_simhash.distance(rep_simhash) / 64
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = big_cid
                
                # 如果找到足够相似的集群，则合并
                if best_match and best_similarity >= self.similarity_threshold:
                    clusters[best_match].append(small_file)
            
            # 移除原小集群
            del clusters[small_cid]
        
        return clusters