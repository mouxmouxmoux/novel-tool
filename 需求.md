# 大规模文本文件去重与分类系统设计

针对您提出的100万个文本文件去重和分类需求，我将设计一个高效、可中断恢复的Python解决方案。

## 系统架构设计

### 1. 整体流程

```
文件扫描 → 编码检测与转换 → 文本清洗 → 特征提取 → 相似度计算 → 聚类分类 → 结果整理
```

### 2. 核心模块

1. **文件处理模块**：负责文件扫描、编码检测和转换
2. **文本清洗模块**：去除广告、评论等非正文内容
3. **特征提取模块**：使用SimHash/MinHash等算法提取文本特征
4. **聚类模块**：基于相似度进行文件聚类
5. **状态管理模块**：支持中断和恢复处理

## 详细实现

### 1. 项目结构

```
text_deduplicator/
├── config.py              # 配置文件
├── main.py                # 主程序入口
├── file_processor.py      # 文件处理模块
├── text_cleaner.py        # 文本清洗模块
├── feature_extractor.py   # 特征提取模块
├── clustering.py          # 聚类模块
├── state_manager.py       # 状态管理模块
├── utils.py               # 工具函数
└── requirements.txt       # 依赖包
```

### 2. 核心代码实现

#### config.py - 配置文件

```python
import os

# 基础配置
SOURCE_DIR = r"D:\path\to\your\txt_files"  # 源文件目录
OUTPUT_DIR = r"D:\path\to\output"          # 输出目录
STATE_FILE = "processing_state.json"       # 状态文件路径
TEMP_DIR = "temp"                          # 临时文件目录

# 处理参数
MIN_DUPLICATE_LENGTH = 500                 # 最小重复字数
SIMILARITY_THRESHOLD = 0.85                # 相似度阈值
BATCH_SIZE = 1000                          # 批处理大小
MAX_WORKERS = 8                            # 最大并行工作数

# 文本清洗参数
MAX_AD_RATIO = 0.3                         # 最大广告内容比例
MIN_CONTENT_RATIO = 0.5                    # 最小正文内容比例
```

#### requirements.txt - 依赖包

```
chardet>=4.0.0
zhconv>=1.4.0
simhash>=0.2.0
numpy>=1.19.0
scikit-learn>=0.24.0
tqdm>=4.60.0
joblib>=1.0.0
```

#### file_processor.py - 文件处理模块

```python
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
```

#### text_cleaner.py - 文本清洗模块

```python
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
        ]
        
        self.comment_patterns = [
            r'读者评论\s*[:：]?\s*.*?(?=\n\n|\n\s*\n|$)',
            r'网友留言\s*[:：]?\s*.*?(?=\n\n|\n\s*\n|$)',
            r'【.*?评论.*?】',
        ]
        
        self.author_note_patterns = [
            r'作者的话\s*[:：]?\s*.*?(?=\n\n|\n\s*\n|$)',
            r'作者留言\s*[:：]?\s*.*?(?=\n\n|\n\s*\n|$)',
            r'【.*?作者.*?】',
        ]
        
        self.garbage_patterns = [
            r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.,，。！？?、；：:;""''()（）【】\[\]{}《》\-\+\*/%=<>~`@#$%^&\|\\]+',  # 非中英文数字标点
            r'\s{2,}',                               # 多个空格
            r'\n{3,}',                               # 多个换行
        ]
        
        # 编译所有正则表达式
        self.compiled_patterns = []
        for patterns in [self.ad_patterns, self.comment_patterns, 
                        self.author_note_patterns, self.garbage_patterns]:
            compiled = [re.compile(p, re.MULTILINE | re.DOTALL) for p in patterns]
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
```

#### feature_extractor.py - 特征提取模块

```python
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
```

#### clustering.py - 聚类模块

```python
import numpy as np
from collections import defaultdict
from datasketch import MinHashLSH
from typing import Dict, List, Tuple, Set

class TextClusterer:
    def __init__(self, similarity_threshold=0.85, min_duplicate_length=500):
        self.similarity_threshold = similarity_threshold
        self.min_duplicate_length = min_duplicate_length
        self.clusters = defaultdict(list)
        self.lsh = MinHashLSH(threshold=1-similarity_threshold, num_perm=128)
    
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
```

#### state_manager.py - 状态管理模块

```python
import json
import os
from typing import Dict, List, Set, Any
from datetime import datetime

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
            'processed_files': set(),
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
        self.state['processed_files'].add(file_path)
    
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
            'processed_files': set(),
            'file_features': {},
            'clusters': {},
            'last_update': datetime.now().isoformat(),
            'current_phase': 'initialized'
        }
        self.save()
```

#### main.py - 主程序入口

```python
import os
import shutil
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import *
from file_processor import FileProcessor
from text_cleaner import TextCleaner
from feature_extractor import FeatureExtractor
from clustering import TextClusterer
from state_manager import StateManager

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('text_deduplicator.log'),
            logging.StreamHandler()
        ]
    )

def process_file_batch(file_batch, file_processor, text_cleaner, feature_extractor, state_manager):
    """处理一批文件"""
    results = []
    
    for original_path, temp_path in file_batch:
        try:
            # 读取清洗后的文本
            with open(temp_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 清洗文本
            cleaned_text, content_ratio = text_cleaner.clean_text(text)
            
            # 检查内容有效性
            if text_cleaner.is_valid_content(text):
                # 提取特征
                features = feature_extractor.extract_features(cleaned_text)
                
                # 保存特征
                state_manager.save_file_features(original_path, features)
                
                results.append((original_path, features))
        except Exception as e:
            logging.error(f"处理文件 {original_path} 失败: {str(e)}")
    
    return results

def organize_results(clusters, output_dir):
    """整理结果到输出目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建集群目录
    for cluster_id, files in clusters.items():
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        
        # 复制文件到集群目录
        for file_path in tqdm(files, desc=f"整理集群 {cluster_id}"):
            try:
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(cluster_dir, file_name)
                shutil.copy2(file_path, dest_path)
            except Exception as e:
                logging.error(f"复制文件 {file_path} 失败: {str(e)}")

def main():
    """主函数"""
    setup_logging()
    logging.info("开始文本去重处理")
    
    # 初始化组件
    state_manager = StateManager(STATE_FILE)
    file_processor = FileProcessor(SOURCE_DIR, TEMP_DIR)
    text_cleaner = TextCleaner()
    feature_extractor = FeatureExtractor()
    clusterer = TextClusterer(SIMILARITY_THRESHOLD, MIN_DUPLICATE_LENGTH)
    
    # 阶段1: 扫描和处理文件
    if state_manager.get_phase() in ['initialized', 'file_processing']:
        state_manager.set_phase('file_processing')
        logging.info("阶段1: 扫描和处理文件")
        
        # 扫描文件
        all_files = file_processor.scan_files()
        logging.info(f"发现 {len(all_files)} 个文本文件")
        
        # 处理文件（编码转换）
        processed_files = file_processor.process_files(all_files, state_manager)
        logging.info(f"处理了 {len(processed_files)} 个文件")
        
        state_manager.save()
    
    # 阶段2: 特征提取
    if state_manager.get_phase() in ['file_processing', 'feature_extraction']:
        state_manager.set_phase('feature_extraction')
        logging.info("阶段2: 特征提取")
        
        # 获取已处理的文件列表
        processed_files = []
        for root, _, files in os.walk(TEMP_DIR):
            for file in files:
                if file.endswith('.txt'):
                    temp_path = os.path.join(root, file)
                    # 找到对应的原始文件路径
                    rel_path = os.path.relpath(temp_path, TEMP_DIR)
                    original_path = os.path.join(SOURCE_DIR, rel_path)
                    if os.path.exists(original_path):
                        processed_files.append((original_path, temp_path))
        
        # 分批处理文件
        batch_size = BATCH_SIZE
        file_batches = [processed_files[i:i + batch_size] 
                       for i in range(0, len(processed_files), batch_size)]
        
        # 使用多线程处理
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for batch in file_batches:
                future = executor.submit(
                    process_file_batch, 
                    batch, 
                    file_processor, 
                    text_cleaner, 
                    feature_extractor,
                    state_manager
                )
                futures.append(future)
            
            # 等待所有任务完成
            for future in tqdm(as_completed(futures), total=len(futures), desc="提取特征"):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"特征提取失败: {str(e)}")
        
        state_manager.save()
    
    # 阶段3: 聚类
    if state_manager.get_phase() in ['feature_extraction', 'clustering']:
        state_manager.set_phase('clustering')
        logging.info("阶段3: 聚类分析")
        
        # 获取文件特征
        file_features = state_manager.get_file_features()
        
        # 将特征从序列化格式转换回对象
        for file_path, features in file_features.items():
            # 重建SimHash
            from simhash import Simhash
            features['simhash'] = Simhash(features['simhash'], 64)
            
            # 重建MinHash
            from datasketch import MinHash
            m = MinHash(num_perm=128)
            m.hashvalues = np.array(features['minhash'])
            features['minhash'] = m
        
        # 执行聚类
        logging.info("使用SimHash进行初步聚类...")
        clusters = clusterer.cluster_by_simhash(file_features)
        
        logging.info("使用MinHash LSH进行精细聚类...")
        clusters = clusterer.cluster_by_minhash_lsh(file_features)
        
        logging.info("合并小集群...")
        clusters = clusterer.merge_small_clusters(clusters, file_features)
        
        # 保存聚类结果
        state_manager.save_clusters(clusters)
        state_manager.save()
        
        logging.info(f"发现 {len(clusters)} 个重复文本集群")
    
    # 阶段4: 整理结果
    if state_manager.get_phase() in ['clustering', 'organizing']:
        state_manager.set_phase('organizing')
        logging.info("阶段4: 整理结果")
        
        clusters = state_manager.get_clusters()
        organize_results(clusters, OUTPUT_DIR)
        
        state_manager.set_phase('completed')
        state_manager.save()
    
    logging.info("处理完成")

if __name__ == "__main__":
    main()
```

## 系统优化与扩展建议

### 1. 性能优化

1. **并行处理**：
   - 使用多进程处理CPU密集型任务（如特征提取）
   - 使用多线程处理IO密集型任务（如文件读写）

2. **内存优化**：
   - 使用生成器而非列表处理大量文件
   - 分批处理特征，避免一次性加载所有特征到内存

3. **算法优化**：
   - 对于超大规模数据，考虑使用近似最近邻搜索(ANN)算法
   - 实现增量式聚类，避免全量重新计算

### 2. 功能扩展

1. **文件名分析**：
   ```python
   def extract_title_from_filename(filename):
       """从文件名提取标题"""
       # 移除章节信息
       title = re.sub(r'第[一二三四五六七八九十百千万\d]+[章节回]', '', filename)
       # 移除作者信息
       title = re.sub(r'作者[:：]?.*?$', '', title)
       # 移除扩展名
       title = os.path.splitext(title)[0]
       return title.strip()
   ```

2. **内容质量评估**：
   ```python
   def assess_content_quality(text):
       """评估内容质量"""
       # 计算有效内容比例
       content_ratio = calculate_content_ratio(text)
       # 检查文本连贯性
       coherence_score = calculate_coherence(text)
       # 综合评分
       quality_score = 0.7 * content_ratio + 0.3 * coherence_score
       return quality_score
   ```

3. **增量处理支持**：
   ```python
   def incremental_clustering(existing_clusters, new_files):
       """增量聚类新文件"""
       # 将新文件与现有集群比较
       for file_path, features in new_files.items():
           best_cluster = find_best_cluster(features, existing_clusters)
           if best_cluster:
               existing_clusters[best_cluster].append(file_path)
           else:
               # 创建新集群
               new_cluster_id = max(existing_clusters.keys()) + 1
               existing_clusters[new_cluster_id] = [file_path]
       return existing_clusters
   ```

## 部署与使用指南

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv text_dedup_env
cd text_dedup_env
Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置修改

编辑`config.py`文件，设置正确的源目录和输出目录。

### 3. 运行程序

```bash
python main.py
```

### 4. 中断与恢复

程序会自动保存处理状态。如果中断，再次运行时会从上次中断的地方继续。

### 5. 结果查看

处理完成后，在输出目录中会看到多个以`cluster_`开头的文件夹，每个文件夹包含一组重复或相似的文本文件。

## 参考资源

1. **中文文本处理**：
   - [zhconv](https://github.com/gumblex/zhconv)：简繁转换库
   - [jieba](https://github.com/fxsjy/jieba)：中文分词库

2. **文本相似度计算**：
   - [SimHash](https://github.com/leonsim/simhash)：局部敏感哈希实现
   - [datasketch](https://github.com/ekzhu/datasketch)：MinHash等算法实现

3. **大规模数据处理**：
   - [Dask](https://dask.org/)：并行计算库
   - [Vaex](https://vaex.io/)：大数据处理库

4. **网络小说处理**：
   - [novel-downloader](https://github.com/0xb82a2/novel-downloader)：网络小说下载工具，包含文本清洗逻辑
   - [webnovel](https://github.com/dipu-bd/lightnovel-crawler)：网络小说爬虫，包含格式化处理

这个系统设计充分考虑了您提出的所有需求，包括大规模文件处理、编码转换、文本清洗、相似度计算、聚类分类以及中断恢复功能。通过分阶段处理和状态管理，系统能够高效稳定地完成100万个文本文件的去重和分类任务。