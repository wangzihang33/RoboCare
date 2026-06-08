import requests
import re
from bs4 import BeautifulSoup
from utils.logger_handler import logger  # 只新增这一行，用于日志

class WebScraper:
    def __init__(self, user_agent='Windows'):
        self.headers = self._get_headers(user_agent)
        logger.info(f"[WebScraper] 初始化，使用 User-Agent: {user_agent}")

    def _get_headers(self, user_agent):
        if user_agent == 'macOS':
            return {
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"macOS"',
            }
        else:
            return {
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
                'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
            }

    def get_webpage_html(self, url):
        response = requests.Response()
        if url.endswith(".pdf"):
            logger.info(f"[WebScraper] 跳过 PDF 链接: {url}")
            return response

        try:
            logger.info(f"[WebScraper] 请求网页: {url}")
            response = requests.get(url, headers=self.headers, timeout=8)
            response.encoding = "utf-8"
            logger.info(f"[WebScraper] 请求成功: {url}，状态码: {response.status_code}")
        except requests.exceptions.Timeout:
            logger.warning(f"[WebScraper] 请求超时: {url}")
        except requests.RequestException as e:
            logger.error(f"[WebScraper] 请求失败: {url}, 错误: {str(e)}")
        return response

    def convert_html_to_soup(self, html):
        html_string = html.text
        soup = BeautifulSoup(html_string, "lxml")
        logger.info(f"[WebScraper] HTML 转换为 Soup 对象成功")
        return soup

    def extract_main_content(self, html_soup, rule=0):
        main_content = []
        tag_rule = re.compile("^(h[1-6]|p|div)" if rule == 1 else "^(h[1-6]|p)")
        for tag in html_soup.find_all(tag_rule):
            tag_text = tag.get_text().strip()
            if tag_text and len(tag_text.split()) > 10:
                main_content.append(tag_text)
        logger.info(f"[WebScraper] 提取正文完成，段落数量: {len(main_content)}")
        return "\n".join(main_content).strip()

    def scrape_url(self, url, rule=0):
        logger.info(f"[WebScraper] 开始抓取 URL: {url}")
        webpage_html = self.get_webpage_html(url)
        soup = self.convert_html_to_soup(webpage_html)
        main_content = self.extract_main_content(soup, rule)
        logger.info(f"[WebScraper] 完成抓取 URL: {url}，正文长度: {len(main_content)}")
        return main_content

# Example usage
if __name__ == "__main__":
    scraper = WebScraper(user_agent='windows')
    test_url = "https://en.wikipedia.org/wiki/Apple_Inc."
    main_content = scraper.scrape_url(test_url)
    print(main_content)