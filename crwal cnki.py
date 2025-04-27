from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException,StaleElementReferenceException, TimeoutException 
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, Manager, cpu_count
from threading import Lock
import re 
import logging
import openpyxl 
import time
import json
import random
import pandas as pd
import os
# 设置日志级别和格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cnki_crawler.log'),
        logging.StreamHandler()
    ]
)

# 目的是为了便于更改配置，避免之前将各个变量定义在函数中用户在更改参数时较为复杂
class Config:
    def __init__(self):
        self.keyword = "多模态融合"
        self.download_path = "F:\\数据爬取与预处理\\大作业\\pdf文件"
        self.chromedriver_path = "F:\\数据爬取与预处理\\chromedriver-win64\\chromedriver.exe"# 替换为你的 chromedriver 路径
        self.max_pages = 100
        self.max_retries = 3
        self.wait_time = 5
        self.pdf_wait_time = 10
        self.crawl_mode = "single"  # "single", "thread",  "process"分别代表简单模式、多线程模式、多进程模式
        self.max_workers = 4
        self.download_pdf = False #是否下载pdf文件
        self.sort_results = True #是否进行排序
        self.sort_key = "cited_by"
        self.sort_ascending = False
        self.output_filename = "papers_info.csv"
        self.paper_types = ["P0209", "P01", "P13"]  # 分别代表不同文献层级如北大核心，AMI，CSSCI等
        self.save_formats = ["csv", "json", "xlsx"] # 代表不同的保存方式
        self.output_directory = r"F:\数据爬取与预处理\大作业"
config = Config()

# 初始化浏览器驱动
def init_driver(download_path):
    # 设置一下pdf文件下载路径
    if download_path is None:
        download_path = config.download_path

    logging.info(f"初始化WebDriver及其文件下载路径: {download_path}")
    # 如果文件路径不存在，创建对应的文件
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    # 配置浏览器选项
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")  # 隐藏自动化标志
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    options.add_argument("--start-maximized")  # 最大化窗口
    # 设置下载路径同时设置一下下载偏好，否则在下载文件时可能会出现限制
    prefs = {"download.default_directory": download_path,
             "download.prompt_for_download": False,
             "download.directory_upgrade": True,
             "safebrowsing.enabled": False}  # 禁用安全检查
    options.add_experimental_option("prefs", prefs)
    service = Service(config.chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)    
    # 隐藏 navigator.webdriver 标志，避免爬取时被检测到selenium
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """
    })
    return driver

# 提取论文信息
def extract_paper_info_with_retry(element, max_retries):
    if max_retries is None:
        max_retries = config.max_retries
    for attempt in range(max_retries):
        try:
            # 每次重试都重新查找子元素
            title_elem = element.find_element(By.CSS_SELECTOR, ".name a.fz14")
            link = title_elem.get_attribute("href")
            title = title_elem.text.strip()
            
            authors_elem = element.find_element(By.CSS_SELECTOR, ".author")
            authors = authors_elem.text.strip()
            
            source_elem = element.find_element(By.CSS_SELECTOR, ".source")
            source = source_elem.text.strip()

            date_elem = element.find_element(By.CSS_SELECTOR, ".date")
            date = date_elem.text.strip()

            cited_by_elem = element.find_element(By.CSS_SELECTOR, ".quote")
            cited_by = cited_by_elem.text.strip().replace("被引量：", "")
            cited_by = int(cited_by) if cited_by.isdigit() else 0
            
            download_elem = element.find_element(By.CSS_SELECTOR, ".download")
            download_count = download_elem.text.strip().replace("下载量：", "")
            download_count = int(download_count) if download_count.isdigit() else 0

            return {
                "title": title,
                "link": link,
                "authors": authors, 
                "source": source,
                "date": date,
                "cited_by": cited_by,           
                "download_count": download_count
            }
            
        except StaleElementReferenceException:
            if attempt == max_retries - 1:
                raise
            logging.warning(f"第 {attempt + 1} 次尝试失败，正在重试...")
            time.sleep(1)
        except Exception as e:
            logging.error(f"提取论文信息失败: {str(e)[:100]}...")
            return None
    return None
#设置一下论文类型选择的复选框
def select_checkbox(value,driver):
    max_retries = config.max_retries
    for attempt in range(max_retries):
        try:
         # 每次重试都重新查找容器
            container = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, '//dd[@field="LYBSM"]'))
                    )             
            checkbox = container.find_element(By.XPATH, f'.//input[@value="{value}"]')
            if not checkbox.is_selected():
                driver.execute_script("arguments[0].click();", checkbox)
                logging.info(f"已点击复选框：{value}")
                time.sleep(1)  # 等待筛选生效
                return True
        except Exception as e:
                logging.warning(f"尝试 {attempt + 1} 失败: {e}")
                if attempt == max_retries - 1:
                    return False
                time.sleep(2)
#尝试在详情页面中找到并点击PDF下载按钮
def attempt_pdf_download(driver, wait_after_click):
    if wait_after_click is None:
        wait_after_click = config.pdf_wait_time
    try:
        # 定位PDF下载按钮
        # 按优先级列出选择器列表，基于提供的HTML结构：
        download_button = None
        # 为了避免页面的代码改变，这里添加了更多选择器
        selectors_to_try = [
            # 1. 首选：通过ID定位（最具体且可靠）
            (By.ID, "pdfDown"),
            # 2. 备选：通过父元素 li 的类名定位（如果ID变化但结构保持不变时有效）
            (By.XPATH, "//li[@class='btn-dlpdf']/a[@id='pdfDown']"),  # 更具体的XPath
            (By.XPATH, "//li[@class='btn-dlpdf']/a"),  # 简单的XPath通过父元素
            # 3. 备选：通过包含的文本内容定位（如果结构发生重大变化时有用）
            (By.XPATH, "//a[contains(normalize-space(), 'PDF下载')]")  # normalize-space处理潜在的多余空格
        ]

        print("尝试定位PDF下载按钮...")
        for by, value in selectors_to_try:
            try:
                # 等待元素出现并可点击
                button = WebDriverWait(driver, 5).until(  # 每个选择器尝试较短的等待时间
                    EC.element_to_be_clickable((by, value))
                )
                # 额外检查按钮是否在页面上可见
                if button.is_displayed():
                    download_button = button
                    logging.info(f"成功找到PDF下载按钮，使用策略: {by}='{value}'")
                    break  # 找到可用按钮后停止搜索
                else:
                    logging.info(f"策略 {by}='{value}' 找到按钮，但按钮不可见，尝试下一个策略...")
            except (NoSuchElementException, TimeoutException):
                logging.info(f"策略 {by}='{value}' 未找到按钮，尝试下一个策略...")
                continue  # 尝试下一个选择器
        # 检查是否成功找到可操作的按钮
        if not download_button:
            logging.info("错误：未能在详细信息页面找到可操作的PDF下载按钮。")
            return False

        # 点击按钮
        logging.info("尝试点击pdf下载按钮")
        # 使用JavaScript点击以确保可靠性，避免覆盖或复杂事件处理程序问题
        driver.execute_script("arguments[0].scrollIntoView(true);", download_button)  # 确保按钮在视图内
        time.sleep(0.5)  # 滚动后短暂暂停再点击
        driver.execute_script("arguments[0].click();", download_button)
        logging.info(f"已点击下载按钮。将等待 {wait_after_click} 秒以允许下载开始...")

        # 重要：等待浏览器的下载过程启动。
        # Selenium不直接管理浏览器下载，此暂停至关重要。
        time.sleep(wait_after_click)
        logging.info("下载等待时间结束。假定下载已开始。")
        return True

    except StaleElementReferenceException:
        logging.error("尝试下载PDF失败：下载按钮元素在操作时失效 (StaleElementReferenceException)。")
        return False
    except Exception as e:
        # 捕获任何其他异常
        logging.error(f"尝试下载PDF时发生未预料的错误: {e}")
        return False
# 提取论文详细信息，包含重试机制。
def extract_detailed_info(driver, link, download_pdf, max_retries):
    if download_pdf is None:
        download_pdf = config.download_pdf
    max_retries = config.max_retries
    original_window = driver.current_window_handle  # 记录原始窗口
    for attempt in range(max_retries):
        try:
            # 在新标签页打开链接
            driver.execute_script(f"window.open('{link}', '_blank');")
            
            # 等待新窗口打开并切换过去
            WebDriverWait(driver, 10).until(
                lambda d: len(d.window_handles) > 1
            )
            driver.switch_to.window(driver.window_handles[-1])
            # 提取摘要
            try:
                abstract = ""
                abstract_elem = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#ChDivSummary"))
                )

                # 检查是否存在“更多”按钮并点击
                try:
                    more_button = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.ID, "ChDivSummaryMore"))
                    )
                    if more_button.is_displayed() and more_button.get_attribute("style") != "display:none":
                        driver.execute_script("arguments[0].click();", more_button)  # 使用 JavaScript 点击
                        # 等待摘要完全展开
                        WebDriverWait(driver, 5).until(
                            EC.invisibility_of_element((By.ID, "ChDivSummaryMore"))
                        )
                except NoSuchElementException:
                    # 没有“更多”按钮，忽略
                    pass
                except Exception as e:
                    print(f"点击“更多”按钮失败: {e}")

                # 提取摘要文本
                abstract_elem = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#ChDivSummary"))
                )
                abstract = abstract_elem.text.strip() if abstract_elem else "无摘要"

            except Exception as e:
                abstract = "无摘要"
                logging.warning(f"提取摘要失败: {e}")

            # 提取关键词
            try:
                keywords_elem = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "body > div.wrapper > div.main > div.container > div > div:nth-child(3) > div:nth-child(4) > p"))
                )
                keywords = ", ".join([keyword.text.strip() for keyword in keywords_elem]) if keywords_elem else "无关键词"
            except Exception as e:
                keywords = "无关键词"
                logging.warning(f"Failed to extract keywords: {e}")

            # 提取基金信息
            try:
                # 根据提供的HTML结构更新选择器： <p class="funds"><span><a>...</a></span></p>
                # 首先尝试使用CSS选择器，然后使用XPath作为备选
                fund_elements = []
                try:
                    #  主要尝试使用CSS选择器
                    fund_elements = WebDriverWait(driver, 7).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.row p.funds span a"))
                    )
                except (NoSuchElementException, TimeoutException):
                    logging.warning("基金信息：CSS 选择器 'div.row p.funds span a' 未找到，尝试 XPath...")
                    # 备选尝试使用XPath
                    try:
                        fund_elements = WebDriverWait(driver, 7).until(
                            EC.presence_of_all_elements_located((By.XPATH, "//p[@class='funds']/span/a"))
                        )
                    except (NoSuchElementException, TimeoutException):
                        logging.warning("基金信息：XPath '//p[@class='funds']/span/a' 也未找到。")
                        fund_elements = [] 
                fund_list = [
                    f.text.strip().rstrip('；').strip() 
                    for f in fund_elements if f.text.strip()
                ]
                fund_projects = " ; ".join(fund_list) if fund_list else "无基金信息"

            except StaleElementReferenceException:
                logging.warning("提取基金信息时元素失效 (Stale)")
                fund_projects = "提取出错 (Stale)"
            except Exception as e_fund:
                logging.warning(f"提取基金信息时发生错误: {e_fund}")
                fund_projects = "提取出错" 

            # 尝试下载PDF
            if download_pdf: 
                # 调用优化后的attempt_pdf_download函数尝试下载PDF
                download_success = attempt_pdf_download(driver, config.pdf_wait_time)
                 # 根据下载结果设置状态信息：成功或失败
                download_status = "尝试成功" if download_success else "尝试失败"
            else:
                download_status = "未尝试"
            driver.close()
            driver.switch_to.window(original_window)

            return {
                "abstract": abstract,
                "keywords": keywords,
                "fund_projects": fund_projects,
                "download_status": download_status
                # 添加其他需要提取的信息
            }
        
        except StaleElementReferenceException as e:
            logging.warning(f"尝试 {attempt + 1}/{max_retries}: 遇到 StaleElementReferenceException，正在重试...")
            time.sleep(2)  # 等待一段时间后重试
        except Exception as e:
            logging.warning(f"提取详细信息失败: {e}")
            if len(driver.window_handles) > 1:
                driver.close()
            driver.switch_to.window(original_window)
            if attempt == max_retries - 1:
                return {
                "abstract": abstract,
                "keywords": keywords,
                "fund_projects": fund_projects,
                "download_status": download_status
                # 添加其他需要提取的信息
            }
            time.sleep(2)
    return  { "abstract": abstract,
              "keywords": keywords,
              "fund_projects": fund_projects,
              "download_status": download_status
                # 添加其他需要提取的信息
            }
#数据清洗与预处理
def clean_paper_data(paper_info):
    if not paper_info:
        return None
    cleaned_info = paper_info.copy()
    for key in ["title", "authors", "source", "abstract", "keywords", "fund_projects"]:
        if key in cleaned_info and isinstance(cleaned_info[key], str):
            cleaned_info[key] = cleaned_info[key].strip()
            cleaned_info[key] = re.sub(r'\s+', ' ', cleaned_info[key])
            if not cleaned_info[key] or cleaned_info[key].lower() in ["无", "n/a", "-", "none"]:
                if key == "abstract": cleaned_info[key] = "无摘要"
                elif key == "keywords": cleaned_info[key] = "无关键词"
                elif key == "fund_projects": cleaned_info[key] = "无基金信息"
                else: cleaned_info[key] = ""
    if "authors" in cleaned_info and cleaned_info["authors"]:
        cleaned_info["authors"] = re.sub(r'\s*[,;]\s*(\[\d+\])?|\s+(\[\d+\])?', '; ', cleaned_info["authors"]).strip('; ')
        cleaned_info["authors"] = re.sub(r'\s*;\s*', '; ', cleaned_info["authors"])
    for key in ["cited_by", "download_count"]:
        if key in cleaned_info:
            if isinstance(cleaned_info[key], str) and cleaned_info[key].isdigit():
                cleaned_info[key] = int(cleaned_info[key])
            elif not isinstance(cleaned_info[key], int):
                cleaned_info[key] = 0
    return cleaned_info

# 爬取 CNKI 数据
def crawl_cnki(keyword, max_pages, download_path):
    if keyword is None:
        keyword = config.keyword
    if max_pages is None:
        max_pages = config.max_pages
    if download_path is None:
        download_path = config.download_path
    driver = init_driver(download_path)
    seen_urls = set()  # 使用URL去重
    papers = []

    try:
        driver.get("https://www.cnki.net/")
        logging.info("进入cnki主页")

        # 搜索操作
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "txt_SearchText"))
        )
        search_box.clear()
        search_box.send_keys(keyword)
        
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "search-btn"))
        )
        try:
            search_button.click()
        except Exception:
            logging.warning("常规点击搜索按钮失败，尝试JS点击...")
            driver.execute_script("arguments[0].click();", search_button)
        logging.info(f"已搜索关键词：{keyword}")
        
        # 等待结果加载
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".result-table-list"))
        )
        
        # 选择文献类型
        # 定义需要点击的复选框的 value 列表
        # 定义需要点击的复选框
        for value in config.paper_types:
            if not select_checkbox(value,driver):
                logging.warning(f"无法选择复选框 {value}，继续执行...")

        current_page = 1
        while current_page <= max_pages:
            logging.info(f"正在爬取第 {current_page} 页...")
            # 滚动加载
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            # 获取当前页所有结果（每次重新查找避免stale）
            results = WebDriverWait(driver, 15).until(
                EC.presence_of_all_elements_located((By.XPATH, '//table[@class="result-table-list"]/tbody/tr'))
            )
            logging.info(f"找到 {len(results)} 条结果")

            # 记录当前页结果数量用于后续staleness判断
            initial_result_count = len(results)
            #由于第0条信息是描述性信息故从1开始循环
            for i in range(1,initial_result_count+1):
                try:
                    # 重新获取当前结果元素（避免stale）
                    current_results = driver.find_elements(By.CSS_SELECTOR, ".result-table-list tr")
                    if i >= len(current_results)+1:
                        break
                        
                    result = current_results[i]
                    paper_info = extract_paper_info_with_retry(result,max_retries=config.max_retries)
                    
                    if paper_info and paper_info["link"] not in seen_urls:
                        seen_urls.add(paper_info["link"])
                        logging.info(f"正在处理第 {i} 条结果: {paper_info['title'][:30]}...")
                        
                        # 提取详细信息
                        detailed_info = extract_detailed_info(driver, paper_info["link"],config.download_pdf,max_retries=config.max_retries)
                        paper_info.update(detailed_info)
                        cleaned_info = clean_paper_data(paper_info)
                        if cleaned_info:
                            papers.append(cleaned_info)
                except StaleElementReferenceException:
                    logging.warning(f"第 {i} 条结果元素失效，跳过...")
                    continue
                except Exception as e:
                    logging.warning(f"处理第 {i} 条结果时出错: {e}")
                    continue
            
            # 翻页操作
            try:
                next_button = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "Page_next_top"))
                )
                
                if not next_button.is_enabled():
                    logging.info("已到最后一页")
                    break
                    
                # 记录当前页的第一个结果元素用于staleness判断
                first_result = driver.find_elements(By.XPATH, '//table[@class="result-table-list"]/tbody/tr')[0]
                
                # 使用JavaScript点击避免元素拦截
                driver.execute_script("arguments[0].click();", next_button)
                logging.info("正在翻页...")
                
                # 等待页面刷新 - 使用多种条件确保完全加载
                WebDriverWait(driver, 20).until(
                    EC.staleness_of(first_result))
                
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".result-table-list"))
                )
                
                current_page += 1
                time.sleep(random.uniform(3, 6))  # 随机等待更长时间
                
            except NoSuchElementException:
                logging.info("已到最后一页")
                break
            except Exception as e:
                logging.error(f"翻页失败: {str(e)[:100]}...")
                break
                
    except Exception as e:
        logging.error(f"爬取过程中发生错误: {e}")
        # 出错时截图
        driver.save_screenshot("error_screenshot.png")
    finally:
        driver.quit()
        return papers
#多线程爬取
def crawl_with_threads(main_driver, keyword, max_pages, download_path, max_workers):
    if keyword is None:
        keyword = config.keyword
    if max_pages is None:
        max_pages = config.max_pages
    if download_path is None:
        download_path = config.download_path
    if max_workers is None:
        max_workers = config.max_workers
    papers = []
    seen_urls = set()
    lock = Lock()
    
    def crawl_page(page):
        """单页面爬取任务"""
        driver = init_driver(download_path)  # 每个线程有自己的driver实例
        try:
            # 执行搜索和筛选（与主线程相同）
            driver.get("https://www.cnki.net/")
            search_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "txt_SearchText"))
            )
            search_box.send_keys(keyword)
            search_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "search-btn"))
            )
            search_button.click()
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".result-table-list"))
            )
            
            # 选择文献类型
            for value in config.paper_types:
                select_checkbox(value, driver)
            
            # 翻到指定页
            if page > 1:
                for _ in range(page - 1):
                    next_button = WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.ID, "Page_next_top"))
                    )
                    next_button.click()
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".result-table-list"))
                    )
                    time.sleep(random.uniform(2, 4))
            
            # 获取当前页结果
            results = WebDriverWait(driver, 15).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".result-table-list tr"))
            )
            
            page_papers = []
            for result in results:
                try:
                    paper_info = extract_paper_info_with_retry(result,max_retries=config.max_retries)
                    if paper_info and paper_info["link"] not in seen_urls:
                        with lock:
                            seen_urls.add(paper_info["link"])
                        
                        detailed_info = extract_detailed_info(driver, paper_info["link"],config.download_pdf,max_retries=config.max_retries)
                        paper_info.update(detailed_info)
                        cleaned_info = clean_paper_data(paper_info)
                        if cleaned_info:
                            page_papers.append(cleaned_info)
                except Exception as e:
                    logging.error(f"处理论文失败: {e}")
                    continue
            return page_papers
                
        finally:
            driver.quit()
    
    # 执行多线程爬取
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(crawl_page, page): page for page in range(1, max_pages + 1)}
        
        for future in as_completed(futures):
            try:
                page_papers = future.result()
                papers.extend(page_papers)
                logging.info(f"爬取第{futures[future]}页成功，共爬取{len(papers)}篇论文")
            except Exception as e:
                logging.error(f"爬取第{futures[future]}页失败：{e}")
    return papers
#多进程爬取
def init_process():
    """初始化进程，设置随机种子"""
    random.seed(os.getpid())
def process_page(args):
    
    driver_path, keyword, page, shared_dict, download_path = args
    try:
        # 每个进程有自己的driver实例
        driver = init_driver(download_path)
        papers = []
        
        # 执行搜索和筛选
        driver.get("https://www.cnki.net/")
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "txt_SearchText"))
        )
        search_box.send_keys(keyword)
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "search-btn"))
        )
        search_button.click()
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".result-table-list"))
        )
        
        # 选择文献类型
        for value in config.paper_types:
            select_checkbox(value, driver)
        
        # 翻到指定页
        if page > 1:
            for _ in range(page - 1):
                next_button = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "Page_next_top"))
                )
                next_button.click()
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".result-table-list"))
                )
                time.sleep(random.uniform(2, 4))
        
        # 获取当前页结果
        results = WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".result-table-list tr"))
        )
        
        for result in results:
            try:
                paper_info = extract_paper_info_with_retry(result,max_retries=config.max_retries)
                if paper_info and paper_info["link"] not in shared_dict:
                    with shared_dict['lock']:
                        shared_dict[paper_info["link"]] = 1  # 标记为已爬取
                    
                    detailed_info = extract_detailed_info(driver, paper_info["link"],config.download_pdf,max_retries=config.max_retries)
                    paper_info.update(detailed_info)
                    cleaned_info = clean_paper_data(paper_info)
                    if cleaned_info:
                        papers.append(cleaned_info)
            except Exception as e:
                logging.error(f"进程 {os.getpid()} 处理论文失败: {e}")
                continue
                
        return (page, papers)
        
    except Exception as e:
        logging.error(f"进程 {os.getpid()} 处理第 {page} 页时出错: {e}")
        return (page, [])
    finally:
        driver.quit()

def crawl_with_processes(keyword, max_pages, download_path, processes=None):
    if keyword is None:
        keyword = config.keyword
    if max_pages is None:
        max_pages = config.max_pages
    if download_path is None:
        download_path = config.download_path
    if processes is None:
        processes = cpu_count()
    # 使用Manager创建共享字典用于去重
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict['lock'] = manager.Lock()  # 用于进程间同步
    
    # 准备参数列表
    driver_path = config.chromedriver_path
    args_list = [(driver_path, keyword, p, shared_dict, download_path) 
                for p in range(1, max_pages + 1)]
    
    papers = []
    try:
        # 创建进程池
        with Pool(processes=processes, initializer=init_process) as pool:
            results = pool.imap_unordered(process_page, args_list)
            
            for page, page_papers in results:
                papers.extend(page_papers)
                print(f"完成第 {page} 页爬取，获取 {len(page_papers)} 篇论文")
                # 定期保存进度
                if page % 2 == 0:
                    save_progress(papers, keyword)
                    
    except KeyboardInterrupt:
        logging.info("用户中断，保存已爬取数据...")
    except Exception as e:
        logging.error(f"多进程爬取出错: {e}")
    finally:
        # 确保最终保存进度
        save_progress(papers, keyword)
        return papers
#保存爬取进度
def save_progress(papers, keyword):
    if keyword is None:
        keyword = config.keyword
    progress_dir = os.path.join(os.getcwd(), keyword)
    if not os.path.exists(progress_dir):
        os.makedirs(progress_dir)
    
    progress_file = os.path.join(progress_dir, "progress.json")
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        logging.info(f"进度已经保存到 {progress_file}")
    except Exception as e:
        logging.error(f"保存进度失败: {e}")

#加载爬取进度
def load_progress(keyword):
    if keyword is None:
        keyword = config.keyword
    progress_file = os.path.join(os.getcwd(), keyword, "progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                papers = json.load(f)
                seen_urls = {p['link'] for p in papers if 'link' in p}
                logging.info(f"从 {progress_file} 恢复进度，已爬取 {len(papers)} 篇论文")
                return papers, seen_urls
        except Exception as e:
            logging.error(f"加载进度失败: {e}")
    return [], set()
# 定义一个函数用于对论文列表进行排序
def sort_papers(papers, sort_key, ascending):
    if not papers:
        return []
        
    if sort_key is None:
        sort_key = config.sort_key
    if ascending is None:
        ascending = config.sort_ascending
        
    # 检查排序键xx'键是否存在于数据中
    if papers and sort_key not in papers[0]:
        logging.warning(f"排序键'{sort_key}'不存在于数据中，无法排序。返回原始列表。")
        return papers
    # 打印排序信息，提示用户按照哪个字段以及升序或降序排序
    logging.info(f"正在对论文列表进行排序，排序键为 '{sort_key}'，排序方式：{'升序' if ascending else '降序'}")

    try:
        # 使用 sorted 函数对论文列表进行排序
        # key 参数使用 lambda 函数处理可能存在的非数值数据
        # 默认情况下，如果数据缺失或类型不匹配，则将其视为 0（如果是整数）或 空字符串（如果是字符串）
        sorted_papers = sorted(
            papers,
            key=lambda x: x.get(sort_key, 0 if isinstance(papers[0].get(sort_key), int) else ""), # Default to 0 or "" based on type
            reverse=not ascending
        )
        logging.info("排序完成。")
        return sorted_papers
    except Exception as e:
        logging.error(f"排序时发生错误: {e}. 返回原始列表。")
        return papers
#将收集到的论文数据保存为指定的一种或多种格式 (csv, json, xlsx)。
def save_data(papers, keyword=None, filename_base=None, output_dir=None, formats=None):
    # 检查是否有数据需要保存
    if not papers:
        logging.warning("没有论文数据可供保存。")
        return

    # 如果未提供参数，则使用配置文件中的默认值
    if keyword is None:
        keyword = config.keyword
    if filename_base is None:
        filename_base = config.output_filename
    if output_dir is None:
        output_dir = config.output_directory
    if formats is None:
        formats = config.save_formats

    # 根据关键词确定具体的输出子目录路径
    keyword_output_dir = os.path.join(output_dir, keyword)

    # --- 创建输出目录 ---
    try:
        # 如果目录不存在，则创建它
        if not os.path.exists(keyword_output_dir):
            os.makedirs(keyword_output_dir)
            logging.info(f"已创建目录：{keyword_output_dir}")
    except OSError as e:
        # 如果创建目录失败，记录错误并返回
        logging.error(f"创建目录 {keyword_output_dir} 失败: {e}")
        return # 没有目录无法继续保存

    # --- 将数据转换为 Pandas DataFrame (CSV 和 Excel 需要) ---
    # 为了避免在循环中重复转换，先进行转换
    try:
        df = pd.DataFrame(papers)
    except Exception as e_df:
        logging.error(f"将论文数据转换为 DataFrame 时失败: {e_df}")
        # 如果无法创建 DataFrame，至少尝试保存为 JSON
        df = None # 标记 DataFrame 创建失败
        if 'json' not in [f.lower().strip() for f in formats]:
            logging.error("DataFrame 创建失败且未配置保存为 JSON，无法保存任何文件。")
            return


    # --- 按照指定的格式循环保存文件 ---
    for fmt in formats:
        fmt = fmt.lower().strip() # 将格式字符串转换为小写并去除首尾空格，以便比较
        # 构造包含扩展名的完整输出文件路径
        output_filename = os.path.join(keyword_output_dir, f"{filename_base}.{fmt}")

        logging.info(f"尝试将数据保存到 {output_filename} (格式: {fmt.upper()})...")
        try:
            if fmt == 'csv':
                if df is not None:
                    # 使用 utf-8-sig 编码确保中文在 Excel 中正确显示
                    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
                    logging.info(f"成功将 {len(papers)} 篇论文信息保存到 CSV 文件: {output_filename}")
                else:
                    logging.error(f"DataFrame 未能创建，无法保存为 CSV 文件: {output_filename}")

            elif fmt == 'json':
                # 对于 JSON，直接保存原始的字典列表通常更好，结构更清晰
                with open(output_filename, 'w', encoding='utf-8') as f:
                    # ensure_ascii=False 保证中文字符正常显示
                    # indent=4 使 JSON 文件格式化，易于阅读
                    json.dump(papers, f, ensure_ascii=False, indent=4)
                logging.info(f"成功将 {len(papers)} 篇论文信息保存到 JSON 文件: {output_filename}")

            elif fmt == 'xlsx':
                # 注意: 保存为 .xlsx 需要 'openpyxl' 库
                if df is not None:
                    try:
                        # engine='openpyxl' 是保存为 .xlsx 格式所必需的
                        df.to_excel(output_filename, index=False, engine='openpyxl')
                        logging.info(f"成功将 {len(papers)} 篇论文信息保存到 Excel 文件: {output_filename}")
                    except ImportError:
                        # 如果缺少 openpyxl 库，则记录错误信息
                        logging.error(f"保存到 Excel (.xlsx) 需要 'openpyxl' 库。")
                        logging.error(f"请使用命令安装: pip install openpyxl")
                    except Exception as ex_excel:
                        # 捕获保存 Excel 时可能出现的其他错误
                         logging.error(f"保存到 Excel 文件 {output_filename} 时失败: {ex_excel}")
                else:
                    logging.error(f"DataFrame 未能创建，无法保存为 Excel 文件: {output_filename}")

            else:
                # 如果指定的格式不受支持，记录警告信息
                logging.warning(f"指定的保存格式不受支持: '{fmt}'。将跳过此格式。")

        except Exception as e_save:
            # 捕获在保存特定格式文件时发生的任何其他错误
            logging.error(f"将数据保存到 {output_filename} (格式: {fmt.upper()}) 时发生错误: {e_save}")


# 主函数
if __name__ == "__main__":
    # 加载已爬进度
    papers, seen_urls = load_progress(config.keyword)
    
    # 选择爬取模式
    if config.crawl_mode == "process":
        # 多进程
        new_papers = crawl_with_processes(
            keyword=config.keyword,
            max_pages=config.max_pages,
            download_path=config.download_path,
            processes=config.max_workers
        )
        papers.extend(new_papers)
    elif config.crawl_mode == "thread":
        # 多线程
        main_driver = init_driver(config.download_path)
        try:
            new_papers = crawl_with_threads(
                main_driver=main_driver,
                keyword=config.keyword,
                max_pages=config.max_pages,
                download_path=config.download_path,
                max_workers=config.max_workers
            )
            papers.extend(new_papers)
        finally:
            main_driver.quit()
    else:
        # 简单模式
        papers = crawl_cnki(
            keyword=config.keyword,
            max_pages=config.max_pages,
            download_path=config.download_path
        )
    
    # 展示结果
    logging.info("爬取结果")
    for paper in papers[:5]:  
        logging.info(paper)
    
    # 爬取结果分类
    if config.sort_results and papers:
        papers = sort_papers(papers,config.sort_key,config.sort_ascending)
        logging.info(f"\n---依据 {config.sort_key}排序前五篇为: ---")
        for i, paper in enumerate(papers[:5]):
            logging.info(f"{i+1}. {paper.get('title', 'N/A')} ({config.sort_key}: {paper.get(config.sort_key, 'N/A')})")

    # --- 最终保存数据 ---
    if papers:
        logging.info("开始将最终结果保存到文件...")
        # 调用新的 save_data 函数，它会根据 config 中的设置保存为多种格式
        save_data(papers) # 使用 config 中的默认参数
    else:
        # 如果没有收集到任何论文，则记录警告
        logging.warning("未能收集到任何论文信息，没有数据可保存。")

    logging.info("脚本执行完毕。")