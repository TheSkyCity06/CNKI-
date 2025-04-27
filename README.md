# CNKI Selenium文献爬取与分析系统

本项目是一个基于 Python 和 Selenium 的工具，旨在自动化地从中国知网 (CNKI) 爬取指定关键词的学术文献元数据及详细信息，并提供一套完整的数据分析流程，用于文献计量学、研究热点追踪、合作网络分析等目的。

## 项目概述

系统分为两大模块：

1.  **文献爬虫模块:** 使用 Selenium 模拟浏览器操作，实现对 CNKI 的智能爬取，具备反检测、并发处理和健壮性设计。
2.  **数据分析模块:** 对爬取到的数据进行清洗、预处理，并执行多维度的统计分析、文本挖掘和可视化。

本项目适用于需要批量获取和分析 CNKI 文献数据的研究人员、学生或数据分析师。

## 主要特性

**爬虫模块:**

* **动态网页处理:** 基于 Selenium，有效处理 JavaScript 动态加载内容。
* **反检测策略:**
    * 禁用 WebDriver 特征 (`--disable-blink-features=AutomationControlled`)。
    * 自定义 User-Agent。
    * 通过 CDP 隐藏 `navigator.webdriver` 属性。
* **并发支持:** 提供多种执行模式 (`crawl_mode`):
    * `single`: 单线程顺序执行。
    * `thread`: 多线程并发执行 (每个线程独立 WebDriver + Lock 同步)。
    * `process`: 多进程并发执行 (每个进程独立 WebDriver + Manager 同步)。
* **健壮性:**
    * 内置**重试机制**处理元素查找失败和 `StaleElementReferenceException`。
    * **进度保存与加载** (`save_progress`, `load_progress`)，支持中断恢复。
    * 全面的 `try-except` 异常处理和详细日志记录 (`logging`)。
    * 随机延时模拟人类行为。
* **信息提取:**
    * 提取文献列表页核心元数据（标题、作者、来源、日期、被引、下载）。
    * 自动打开详情页提取摘要、关键词、基金信息。
    * **可选 PDF 下载**功能 (`download_pdf` 开关)。
* **灵活性:**
    * 通过 `Config` 类集中配置所有参数。
    * 支持筛选文献类型（如北大核心、CSSCI 等）。
    * 支持结果排序 (`sort_papers`)。
    * 支持多种输出格式 (`save_formats`: CSV, JSON, XLSX)。
    * 输出文件按关键词存于独立子目录。

**分析模块:**

* **数据预处理:** 自动清洗数值、提取年份、分割作者/关键词列表、处理缺失值。
* **基础统计:** 计算被引/下载量描述性统计。
* **趋势分析:** 绘制年度发文量趋势图。
* **核心识别:**
    * Top N 作者、关键词、发表来源的可视化（条形图）。
    * 来源（期刊/会议）影响力分析（平均被引/下载量）。
* **合作分析:** 作者数量分布、合作强度分析、作者合作网络可视化 (`NetworkX`)。
* **热点挖掘:**
    * 关键词/摘要词云 (`WordCloud`, `jieba`)。
    * 关键词共现网络可视化 (`NetworkX`)。
    * 关键词年度频率动态趋势图。
* **主题建模:** 使用 TF-IDF 和 LDA (`scikit-learn`) 对摘要进行主题分析，识别潜在研究方向。
* **相关性分析:** 计算数值特征（被引、下载、年份、作者数）间的相关性并用热力图展示。
* **可视化:** 使用 Matplotlib 和 Seaborn 生成多种图表，支持中文显示（需字体文件）。
* **配置化:** 通过 `AnalysisConfig` 类管理分析参数。

## 系统要求

* Python 3.x
* **必要的 Python 库:** (建议使用 `pip install -r requirements.txt` 安装，如果提供了该文件)
    * `selenium`
    * `pandas`
    * `openpyxl` (用于读写 Excel)
    * `numpy`
    * `matplotlib`
    * `seaborn`
    * `wordcloud`
    * `jieba`
    * `networkx`
    * `scikit-learn`
    * `Pillow` (PIL Fork, WordCloud 可能需要)
* **ChromeDriver:** 版本需与您安装的 Chrome 浏览器版本匹配。 [下载地址](https://googlechromelabs.github.io/chrome-for-testing/) 或 [淘宝镜像](https://registry.npmmirror.com/binary.html?path=chromedriver/)
* **中文字体文件:** 用于词云和图表中文显示，例如 `simhei.ttf` (黑体)。请确保文件存在且路径配置正确。
* **停用词文件:** (可选) 用于主题建模和词云，例如 `stopwords.txt`。

## 安装与设置

1.  **获取代码:** 克隆本仓库或下载代码文件。
2.  **安装依赖:**
    ```bash
    pip install selenium pandas openpyxl numpy matplotlib seaborn wordcloud jieba networkx scikit-learn Pillow
    ```
    (或者，如果提供了 `requirements.txt` 文件: `pip install -r requirements.txt`)
3.  **准备 ChromeDriver:** 下载与您的 Chrome 版本匹配的 ChromeDriver，将其放置在系统 PATH 中，或者记下其绝对路径。
4.  **准备字体和停用词文件:** 将 `simhei.ttf` (或其他中文字体) 和 `stopwords.txt` 文件放置在项目目录下，或方便引用的位置。

## 配置说明

在运行脚本前，需要根据您的环境修改配置。

**爬虫配置 (假设在 `crawler_script.py` 中):**

打开爬虫脚本文件，找到 `Config` 类实例化的部分，修改以下关键参数：

```python
config = Config()
config.keyword = "多模态融合"  # ***【必须修改】*** 设置您要爬取的关键词
config.chromedriver_path = "F:\\path\\to\\your\\chromedriver.exe"  # ***【必须修改】*** ChromeDriver 的绝对路径
config.download_path = "F:\\path\\to\\save\\pdfs" # 如果启用PDF下载，设置PDF保存路径
config.output_directory = r"F:\path\\to\\save\\results" # 设置爬取结果(CSV/JSON/XLSX)的基础目录
config.crawl_mode = "single"  # 可选 "single", "thread", "process"
config.max_pages = 10         # 设置最大爬取页数
config.download_pdf = False   # 是否下载PDF (True/False)
config.save_formats = ["csv", "xlsx"] # 输出文件格式列表
# ... 其他参数按需修改 (如 max_retries, wait_time, paper_types, sort_results 等)
