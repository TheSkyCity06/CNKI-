import pandas as pd
import numpy as np
import re
from collections import Counter
from functools import reduce, partial
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud
from PIL import Image
import jieba
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import itertools

# --- 基本设置 ---
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 配置 ---
class AnalysisConfig:
    def __init__(self):
        self.keyword = "多模态融合"
        self.input_filename = "papers_info.csv.csv"
        self.base_directory = r"F:\数据爬取与预处理\大作业"
        self.year_extraction_pattern = r'(\d{4})'
        self.author_separator = r'\s*;\s*'
        self.keyword_separator = r'\s*[;,]\s*|\s+'
        self.top_n_results = 15
        self.network_node_threshold = 5
        self.network_top_n = 30
        self.num_topics = 5
        self.stopwords_path = "stopwords.txt"
        self.top_n_keywords_trend = 10

analysis_config = AnalysisConfig()

# --- 纯辅助函数 ---

def safe_to_numeric(series):
    """将 Pandas Series 转换为数值类型，无法转换的值变为 NaN。"""
    return pd.to_numeric(series, errors='coerce')

def fill_na_numeric(series, fill_value = 0):
    """使用指定值填充数值 Series 中的 NaN。"""
    return series.fillna(fill_value).astype(int)

def extract_year(date_str, pattern):
    """从字符串中提取第一个四位数的年份。"""
    if not isinstance(date_str, str):
        return None
    match = re.search(pattern, date_str)
    return int(match.group(1)) if match else None

def split_and_clean(text, separator_pattern):
    """根据正则表达式模式分割字符串并清理空白。"""
    if not isinstance(text, str) or pd.isna(text) or text.lower() in ["无", "n/a", "-", "none", "无关键词", "无作者", "未知作者"]:
        return []
    text = re.sub(r'\s*\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    items = re.split(separator_pattern, text)
    return [item.strip() for item in items if item and item.strip() and len(item.strip()) > 1]

def flatten_list_of_lists(list_of_lists):
    """将包含列表的列表展平成单个列表。"""
    return [item for sublist in list_of_lists for item in sublist]

def count_items(items):
    """计算列表中各项的频率。"""
    return dict(Counter(items))

def get_top_n_items(item_counts, n):
    """从频率字典中获取前 N 个条目。"""
    return sorted(item_counts.items(), key=lambda item: item[1], reverse=True)[:n]

def load_stopwords(filepath):
    """从文件加载停用词列表。"""
    stopwords = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                stopwords = [line.strip() for line in f if line.strip()]
            print(f"成功从 {filepath} 加载 {len(stopwords)} 个停用词。")
        except Exception as e:
            print(f"加载停用词文件 {filepath} 时出错: {e}。将使用空列表。")
    else:
        print(f"警告：停用词文件 {filepath} 未找到。将不使用停用词。")
    stopwords.extend(['研究', '分析', '基于', '模型', '方法', '应用', '探讨', '实现', '设计', '系统'])
    return list(set(stopwords))

# --- 数据预处理流程函数 ---
def preprocess_data(df, config):
    """对 DataFrame 应用一系列预处理步骤。"""
    processed_df = df.copy()
    processed_df['cited_by'] = fill_na_numeric(safe_to_numeric(processed_df['cited_by']))
    processed_df['download_count'] = fill_na_numeric(safe_to_numeric(processed_df['download_count']))
    year_extractor = partial(extract_year, pattern=config.year_extraction_pattern)
    processed_df['year'] = processed_df['date'].apply(year_extractor)
    processed_df = processed_df.dropna(subset=['year'])
    processed_df['year'] = processed_df['year'].astype(int)
    author_splitter = partial(split_and_clean, separator_pattern=config.author_separator)
    processed_df['author_list'] = processed_df['authors'].apply(author_splitter)
    keyword_splitter = partial(split_and_clean, separator_pattern=config.keyword_separator)
    processed_df['keyword_list'] = processed_df['keywords'].apply(keyword_splitter)
    processed_df['abstract'] = processed_df['abstract'].astype(str).str.strip().fillna("无摘要")
    processed_df['title'] = processed_df['title'].astype(str).str.strip()
    processed_df['source'] = processed_df['source'].astype(str).str.strip()
    processed_df['num_authors'] = processed_df['author_list'].apply(len)
    print("数据预处理完成。")
    return processed_df

# --- 原有分析函数 ---
def analyze_basic_stats(df):
    """计算数值字段的描述性统计信息。"""
    print("\n--- 基础统计 (被引量和下载量) ---")
    stats = df[['cited_by', 'download_count']].describe()
    print(stats)
    return stats

def analyze_publication_trends(df, config):
    """分析每年的论文发表数量。"""
    print("\n--- 发表趋势 (每年论文数) ---")
    trends = df['year'].value_counts().sort_index()
    print(trends)
    plt.figure(figsize=(12, 6))
    trends.plot(kind='line', marker='o')
    plt.title(f'"{config.keyword}" 相关论文发表趋势')
    plt.xlabel('年份')
    plt.ylabel('论文数量')
    plt.grid(True); plt.tight_layout(); plt.show()
    return trends

def analyze_top_items(df, column, item_name, config):
    """通用函数，用于查找列表列中的前 N 个项目 (例如，作者、关键词)。"""
    print(f"\n--- Top {config.top_n_results} {item_name} ---")
    all_items = flatten_list_of_lists(df[column].tolist())
    item_counts = count_items(all_items)
    top_items = get_top_n_items(item_counts, config.top_n_results)
    if not top_items: print(f"未找到有效的 {item_name}。"); return []
    for item, count in top_items: print(f"{item}: {count}")
    items, counts = zip(*top_items)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(counts), y=list(items), palette='viridis', orient='h')
    plt.title(f'Top {config.top_n_results} {item_name} ("{config.keyword}")'); plt.xlabel('频次'); plt.ylabel(item_name)
    plt.tight_layout(); plt.show()
    return top_items

def analyze_top_sources(df, config):
    """分析排名前 N 的发表来源。"""
    print(f"\n--- Top {config.top_n_results} 发表来源 ---")
    source_counts = df['source'].value_counts()
    top_sources = source_counts.head(config.top_n_results).reset_index().values.tolist()
    if not top_sources: print("未找到来源信息。"); return []
    for source, count in top_sources: print(f"{source}: {count}")
    sources, counts = zip(*top_sources)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(counts), y=list(sources), palette='magma', orient='h')
    plt.title(f'Top {config.top_n_results} 发表来源 ("{config.keyword}")'); plt.xlabel('论文数量'); plt.ylabel('来源')
    plt.tight_layout(); plt.show()
    return top_sources

def analyze_collaboration(df, config):
    """分析作者合作模式。"""
    print("\n--- 合作分析 ---")
    valid_papers = df[df['num_authors'] > 0]
    if valid_papers.empty: 
        print("未找到包含作者信息的论文进行合作分析。")
        return {}
    
    # 计算统计量
    avg_authors = valid_papers['num_authors'].mean()
    median_authors = valid_papers['num_authors'].median()
    max_authors = valid_papers['num_authors'].max()
    single_author_papers = valid_papers[valid_papers['num_authors'] == 1].shape[0]
    multi_author_papers = valid_papers[valid_papers['num_authors'] > 1].shape[0]
    total_valid_papers = valid_papers.shape[0]
    single_author_ratio = single_author_papers / total_valid_papers if total_valid_papers else 0
    
    results = {
        "Average Authors per Paper": avg_authors,
        "Median Authors per Paper": median_authors,
        "Max Authors in a Paper": max_authors,
        "Single Author Paper Ratio": single_author_ratio
    }
    
    print(f"平均每篇论文作者数: {avg_authors:.2f}")
    print(f"每篇论文作者数中位数: {median_authors}")
    print(f"单篇论文最大作者数: {max_authors}")
    print(f"单作者论文比例: {single_author_ratio:.2%}")
    
    # 绘制改进后的直方图
    plt.figure(figsize=(12, 7))
    
    # 处理极端值 - 设置合理的显示范围
    data = valid_papers['num_authors']
    upper_limit = np.percentile(data, 99)  # 显示99%的数据
    filtered_data = data[data <= upper_limit]
    
    # 智能分箱
    if max_authors <= 10:
        bins = range(1, max_authors + 2)
    else:
        bins = min(20, max_authors)
    
    ax = sns.histplot(filtered_data, bins=bins, kde=False, color='steelblue')
    
    # 添加数值标签
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 3),
                        textcoords='offset points')
    
    plt.title(f'每篇论文作者数量分布 ("{config.keyword}")', pad=20)
    plt.xlabel('作者数量', labelpad=10)
    plt.ylabel('论文篇数', labelpad=10)
    
    # 优化刻度
    if max_authors <= 20:
        plt.xticks(range(1, max_authors + 1))
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 添加说明文本
    if max_authors > upper_limit:
        plt.text(0.95, 0.95, 
                f"注: 已排除少量作者数>{upper_limit}的极端值", 
                transform=ax.transAxes,
                ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()
    return results

def generate_word_cloud(text, title, mask_path = None, font_path = 'simhei.ttf'):
    """生成中文词云图"""
    if not text or text.isspace(): print(f"文本为空，无法为 '{title}' 生成词云。"); return
    try:
        if not os.path.exists(font_path):
             print(f"警告: 字体文件 '{font_path}' 未找到，词云可能无法正确显示中文。请确保字体文件路径正确。")

        wordlist = jieba.lcut(text)
        text_processed = " ".join(wordlist)
        wordcloud_params = {'font_path': font_path, 'background_color': 'white', 'max_words': 200, 'max_font_size': 100, 'width': 800, 'height': 600, 'random_state': 42}
        mask = None
        if mask_path and os.path.exists(mask_path):
            try: mask = np.array(Image.open(mask_path)); wordcloud_params['mask'] = mask
            except Exception as e_mask: print(f"加载掩码图片 {mask_path} 失败: {e_mask}")

        wordcloud = WordCloud(**wordcloud_params).generate(text_processed)
        plt.figure(figsize=(12, 8)); plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=16); plt.axis('off'); plt.tight_layout(); plt.show()
    except Exception as e_wc:
        print(f"为 '{title}' 生成词云时出错: {e_wc}")


def analyze_keyword_frequency(df, config):
    """生成关键词词云"""
    print("\n--- 关键词词云 ---")
    all_keywords = flatten_list_of_lists(df['keyword_list'].tolist())
    if not all_keywords: print("没有可分析的关键词数据。"); return
    keyword_text = " ".join(all_keywords)
    generate_word_cloud(keyword_text, f'"{config.keyword}"相关研究关键词词云')

def analyze_abstract_wordcloud(df, config):
    """生成摘要词云"""
    print("\n--- 摘要词云 ---")
    all_abstracts = " ".join(df['abstract'].astype(str).tolist())
    if not all_abstracts or len(all_abstracts) < 10: print("没有足够的摘要文本进行分析。"); return
    generate_word_cloud(all_abstracts, f'"{config.keyword}"相关研究摘要词云')


# --- 深度分析函数 ---

def analyze_network(df, column, item_name, config):
    """通用函数，用于分析和可视化合作/共现网络。"""
    print(f"\n--- {item_name} 合作/共现网络分析 ---")
    pairs = []
    for item_list in df[column]:
        if len(item_list) >= 2:
            pairs.extend(list(itertools.combinations(sorted(item_list), 2)))

    if not pairs: print(f"未能生成足够的 {item_name} 对来构建网络。"); return
    pair_counts = Counter(pairs)
    print(f"共找到 {len(pair_counts)} 组不同的 {item_name} 合作/共现关系。")
    G = nx.Graph()
    for (item1, item2), weight in pair_counts.items():
        G.add_edge(item1, item2, weight=weight)

    if G.number_of_nodes() == 0: print("未能构建有效的网络图。"); return
    print(f"网络图已构建: {G.number_of_nodes()} 个节点 (总 {item_name}), {G.number_of_edges()} 条边 (关系)")
    nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree < config.network_node_threshold]
    G.remove_nodes_from(nodes_to_remove)
    if G.number_of_nodes() > 0:
        try: # 添加 try-except 处理空图或只有一个组件的情况
             largest_cc = max(nx.connected_components(G), key=len)
             G = G.subgraph(largest_cc).copy()
        except ValueError: # 如果没有组件 (空图)
             print("过滤后网络图为空。")
             return
    else:
         print("过滤后网络图为空。")
         return

    if G.number_of_nodes() == 0: print("过滤后网络图为空，无法可视化。"); return
    elif G.number_of_nodes() > 300: print(f"警告：过滤后的网络图仍有 {G.number_of_nodes()} 个节点，可视化可能非常缓慢或混乱。")
    print(f"过滤和简化后网络图: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")

    plt.figure(figsize=(16, 16))
    try: # 添加 try-except 保证布局算法成功
        pos = nx.kamada_kawai_layout(G)
    except nx.NetworkXError as e_layout:
        print(f"Kamada-Kawai 布局失败 ({e_layout})，尝试 Spring 布局...")
        try:
             pos = nx.spring_layout(G, k=0.3, iterations=50)
        except Exception as e_layout2:
             print(f"Spring 布局也失败 ({e_layout2})，无法进行可视化。")
             plt.close() # 关闭图形窗口
             return

    node_degrees = dict(G.degree())
    node_sizes = [v * 20 + 50 for v in node_degrees.values()]
    edge_weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='SimHei')
    plt.title(f'{item_name} 合作/共现网络图 ("{config.keyword}")', fontsize=16)
    plt.axis('off'); plt.tight_layout(); plt.show()


def analyze_topic_modeling(df, config):
    """使用LDA对摘要进行主题建模。"""
    print(f"\n--- 摘要主题建模 (LDA, {config.num_topics} 个主题) ---")
    abstracts = df['abstract'].dropna().astype(str).tolist()
    if not abstracts: print("没有有效的摘要数据进行主题建模。"); return
    stopwords = load_stopwords(config.stopwords_path)
    tokenized_abstracts = [" ".join(jieba.lcut(text)) for text in abstracts]
    try:
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stopwords)
        tfidf_matrix = vectorizer.fit_transform(tokenized_abstracts)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError as e_tfidf: print(f"TF-IDF向量化失败: {e_tfidf}。"); return
    except Exception as e: print(f"TF-IDF向量化时发生未知错误: {e}"); return
    if tfidf_matrix.shape[1] == 0: print("向量化后没有有效的词语特征，无法进行主题建模。"); return

    print("正在训练 LDA 模型...")
    lda = LatentDirichletAllocation(n_components=config.num_topics, max_iter=10, learning_method='online', learning_offset=50., random_state=42)
    try:
        lda.fit(tfidf_matrix)
    except Exception as e_lda: print(f"LDA 模型训练失败: {e_lda}"); return

    print("\n各主题下的主要关键词:")
    n_top_words = 10
    for topic_idx, topic in enumerate(lda.components_):
        top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        print(f"主题 #{topic_idx + 1}: {', '.join(top_words)}")


def analyze_keyword_trends(df, config):
    """分析关键词随时间的变化趋势。"""
    print(f"\n--- Top {config.top_n_keywords_trend} 关键词年度趋势 ---")
    all_keywords = flatten_list_of_lists(df['keyword_list'].tolist())
    if not all_keywords: print("无关键词数据。"); return
    keyword_counts = count_items(all_keywords)
    top_keywords_overall = get_top_n_items(keyword_counts, config.top_n_keywords_trend)
    keywords_to_track = [kw[0] for kw in top_keywords_overall]
    if not keywords_to_track: print("未能确定要追踪的关键词。"); return
    print(f"将追踪以下关键词的年度频率: {', '.join(keywords_to_track)}")

    yearly_keyword_freq = {}
    valid_years = sorted(df['year'].unique())
    for year in valid_years:
        yearly_df = df[df['year'] == year]
        year_keywords = flatten_list_of_lists(yearly_df['keyword_list'].tolist())
        year_counts = count_items(year_keywords)
        yearly_keyword_freq[year] = {kw: year_counts.get(kw, 0) for kw in keywords_to_track}

    trends_df = pd.DataFrame.from_dict(yearly_keyword_freq, orient='index')
    trends_df.sort_index(inplace=True)
    if trends_df.empty: print("未能生成关键词趋势数据。"); return # 检查是否为空

    plt.figure(figsize=(15, 8))
    for keyword in trends_df.columns:
        plt.plot(trends_df.index, trends_df[keyword], marker='o', linestyle='-', label=keyword)

    plt.title(f'Top {config.top_n_keywords_trend} 关键词年度出现频率趋势 ("{config.keyword}")')
    plt.xlabel('年份'); plt.ylabel('年度出现次数')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True); plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.show()

def analyze_correlations(df, config):
    """计算并可视化数值特征之间的相关性。"""
    print("\n--- 数值特征相关性分析 ---")
    correlation_cols = ['cited_by', 'download_count', 'year', 'num_authors']
    valid_cols = [col for col in correlation_cols if col in df.columns]
    if len(valid_cols) < 2: print("数据中不足两个有效的数值列进行相关性分析。"); return
    correlation_matrix = df[valid_cols].corr()
    print("相关性矩阵:"); print(correlation_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'数值特征相关性热力图 ("{config.keyword}")')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.show()

def analyze_source_impact(df, config):
     """按来源分析论文数量和平均影响力（被引/下载）。"""
     print(f"\n--- Top {config.top_n_results} 发表来源影响力分析 ---")
     if 'source' not in df.columns or df['source'].isnull().all(): print("数据中缺少有效的'source'列。"); return
     source_stats = df.groupby('source').agg(
         paper_count=('title', 'count'), avg_citations=('cited_by', 'mean'),
         avg_downloads=('download_count', 'mean'), total_citations=('cited_by', 'sum'),
     ).reset_index()
     top_sources_by_count = source_stats.sort_values('paper_count', ascending=False).head(config.top_n_results)
     print("\n按论文数量排名的 Top 来源及其平均影响力:")
     print(top_sources_by_count[['source', 'paper_count', 'avg_citations', 'avg_downloads']].round(2))

     # 确保有数据再绘图
     if not top_sources_by_count.empty:
        plt.figure(figsize=(12, 8))
        plot_data_cite = top_sources_by_count.sort_values('avg_citations', ascending=False)
        sns.barplot(x='avg_citations', y='source', data=plot_data_cite, palette='crest', orient='h')
        plt.title(f'Top {config.top_n_results} 来源的平均被引次数 ("{config.keyword}")'); plt.xlabel('平均被引次数'); plt.ylabel('来源')
        plt.tight_layout(); plt.show()

        plt.figure(figsize=(12, 8))
        plot_data_down = top_sources_by_count.sort_values('avg_downloads', ascending=False)
        sns.barplot(x='avg_downloads', y='source', data=plot_data_down, palette='flare', orient='h')
        plt.title(f'Top {config.top_n_results} 来源的平均下载次数 ("{config.keyword}")'); plt.xlabel('平均下载次数'); plt.ylabel('来源')
        plt.tight_layout(); plt.show()
     else:
        print("未能生成来源影响力数据用于绘图。")

# --- 主执行流程 ---
def load_data(filepath):
    """从 CSV 文件加载数据。"""
    if not os.path.exists(filepath):
        print(f"错误：文件未找到 {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        print(f"成功从 {filepath} 加载数据。 维度: {df.shape}")
        required_cols = ['title', 'authors', 'source', 'date', 'cited_by', 'download_count', 'abstract', 'keywords']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols: print(f"警告：输入文件中缺少以下必需列: {', '.join(missing_cols)}")
        return df
    except Exception as e:
        print(f"加载 CSV 文件 {filepath} 时出错: {e}")
        return None

if __name__ == "__main__":
    input_file = os.path.join(
        analysis_config.base_directory,
        analysis_config.keyword,
        analysis_config.input_filename
    )
    raw_df = load_data(input_file)

    if raw_df is not None:
        processed_df = preprocess_data(raw_df, analysis_config)
        if not processed_df.empty:
            analyze_basic_stats(processed_df)
            analyze_publication_trends(processed_df, analysis_config)
            top_authors = analyze_top_items(processed_df, 'author_list', '作者', analysis_config)
            top_keywords = analyze_top_items(processed_df, 'keyword_list', '关键词', analysis_config)
            analyze_top_sources(processed_df, analysis_config)
            analyze_collaboration(processed_df, analysis_config)
            analyze_keyword_frequency(processed_df, analysis_config)
            analyze_abstract_wordcloud(processed_df, analysis_config)

            print("\n" + "="*20 + " 新增分析功能 " + "="*20)
            analyze_correlations(processed_df, analysis_config)
            analyze_source_impact(processed_df, analysis_config)
            analyze_keyword_trends(processed_df, analysis_config)
            analyze_topic_modeling(processed_df, analysis_config)
            analyze_network(processed_df, 'author_list', '作者', analysis_config)
            analyze_network(processed_df, 'keyword_list', '关键词', analysis_config)

            print("\n--- 分析总结 ---")
            print(f"分析了关于关键词 '{analysis_config.keyword}' 的 {processed_df.shape[0]} 篇有效论文。")
            # ... (其他总结信息)
        else:
             print("预处理后没有有效数据可供分析。")
        print("\n分析结束。")