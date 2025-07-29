import os
import shutil
import logging
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import *
from file_processor import FileProcessor
from text_cleaner import TextCleaner
from feature_extractor import FeatureExtractor
from clustering import TextClusterer
from state_manager import StateManager
from simhash import Simhash
from datasketch import MinHash

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
            features['simhash'] = Simhash(features['simhash'], 64)
            
            # 重建MinHash
            m = MinHash(num_perm=128)
            m.hashvalues = np.array(features['minhash'])
            features['minhash'] = m
        
        # 执行聚类
        logging.info("使用文件名相似性进行初步聚类...")
        filename_clusters = clusterer.cluster_by_filename(file_features)
        logging.info(f"基于文件名相似性发现 {len(filename_clusters)} 个集群")
        
        logging.info("使用SimHash进行初步聚类...")
        clusters = clusterer.cluster_by_simhash(file_features)
        
        logging.info("使用MinHash LSH进行精细聚类...")
        clusters = clusterer.cluster_by_minhash_lsh(file_features)
        
        logging.info("合并小集群...")
        clusters = clusterer.merge_small_clusters(clusters, file_features)
        
        # 合并文件名聚类结果和内容聚类结果
        # 这里我们优先考虑文件名聚类结果，然后是内容聚类结果
        all_cluster_id = max([max(clusters.keys()) if clusters else 0, 
                              max(filename_clusters.keys()) if filename_clusters else 0]) + 1
        
        # 将文件名聚类结果添加到最终结果中（避免重复）
        file_in_clusters = set()
        for files in clusters.values():
            file_in_clusters.update(files)
        
        for cid, files in filename_clusters.items():
            # 检查这些文件是否已经存在于内容聚类中
            new_files = [f for f in files if f not in file_in_clusters]
            if new_files:
                clusters[all_cluster_id] = new_files
                all_cluster_id += 1
                file_in_clusters.update(new_files)
        
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