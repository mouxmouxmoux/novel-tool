import os

# 基础配置
SOURCE_DIR = r"z:\女频"  # 源文件目录
OUTPUT_DIR = r"z:\女频\dealed"          # 输出目录
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