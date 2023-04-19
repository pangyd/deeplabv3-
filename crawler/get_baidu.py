import random
import time
import requests
import sys

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import goto
from dominate.tags import label

from goto import with_goto

page_num = 10  # 默认爬取页数


# 随机ip代理获取
# PROXY_POOL_URL = 'http://localhost:5555/random'


def get_proxy():
    try:
        # response = requests.get(PROXY_POOL_URL)
        # if response.status_code == 200:
        #     return response.text
        list = ['222.184.59.8:808', '47.91.166.84:80', '111.202.247.50:8080', '47.99.72.176:8080', '59.55.120.168:80',
                '47.94.215.245:80', '39.105.63.222:80', '47.104.107.19:8080', '183.232.231.133:80', '117.185.16.226:80',
                '47.114.129.251:80', '39.104.68.232:8080', '117.185.17.177:80', '127.0.0.1:80', '127.8.9.170:80',
                '85.14.243.31:80', '89.221.223.234:88', '116.62.226.124:80', '173.208.235.202:88', '46.101.238.238:80',
                '164.83.111.79:8080', '134.119.223.242:80', '46.105.51.183:80', '1.234.27.121:80', '1.234.45.130:80',
                '178.248.214.130:80', '206.189.156.61:80', '149.28.147.98:80', '213.136.79.111:80',
                '132.148.148.221:80', '8.210.19.205:80', '149.86.225.105:80', '148.72.152.156:80', '35.227.217.65:80',
                '203.154.71.139:81', '13.51.131.184:80', '67.207.70.81:80', '5.189.128.171:80', '60.205.132.71:80',
                '143.110.221.87:80', '5.196.75.109:80', '159.65.219.73:80', '209.181.233.7:80', '159.65.228.74:80',
                '164.83.224.106:8080', '47.93.232.37:80', '195.78.212.147:80', '164.132.112.237:80',
                '108.61.149.126:80', '46.105.122.177:80', '159.192.104.80:80', '15.185.180.14:80', '85.214.136.215:80',
                '34.254.107.230:80', '212.162.68.121:80', '165.22.255.81:80', '173.254.206.34:80', '121.13.239.139:808',
                '185.78.196.46:80', '188.240.71.213:80', '3.128.221.102:80', '202.31.11.166:80', '161.35.97.206:80',
                '52.202.134.188:80', '54.159.141.165:80', '206.125.41.144:80', '31.220.63.167:80', '47.95.205.25:80',
                '95.84.145.67:81', '51.195.137.123:80', '112.125.88.226:80', '202.165.47.90:203', '106.14.126.247:8080',
                '190.1.201.58:80', '82.220.38.70:80', '36.89.218.67:85', '36.37.160.242:80', '45.66.54.20:80',
                '201.253.99.243:81', '167.71.205.73:81', '51.81.83.82:80', '3.230.134.131:80', '198.144.149.82:82',
                '218.78.22.146:83', '124.204.33.162:89', '95.217.133.239:80', '120.26.123.95:8010',
                '219.136.245.77:1080', '118.70.144.77:80', '123.128.12.87:30001', '8.218.80.41:59394',
                '201.236.224.123:999', '203.189.199.18:8888', '116.58.251.67:80', '113.28.90.67:80',
                '47.243.74.226:59394', '88.99.25.49:80', '47.242.86.153:59394', '124.65.144.38:7302',
                '185.233.193.8:8080', '140.246.149.224:8888', '113.125.189.71:8888', '103.143.196.50:8080',
                '8.210.219.124:59394', '171.97.12.180:8080', '103.165.250.2:8080', '183.89.68.110:8080',
                '47.243.90.97:8080', '43.132.166.30:59394', '14.207.127.175:88', '14.207.127.175:8080',
                '45.32.85.10:3128', '175.10.223.95:8060', '182.84.66.207:37824']
        a = random.choice(list)
        return a
    except ConnectionError:
        return None


def get_baidu(key):
    process_start_time = time.time()

    # 获取一个代理ip
    my_ip = get_proxy()

    print(my_ip)

    # 不打开浏览器
    option = webdriver.EdgeOptions()
    option.add_argument("headless")
    option.add_argument('--disable-blink-features=AutomationControlled')
    # option.add_argument("--proxy-server=http://" + my_ip)

    try:
        browser = webdriver.Edge(options=option)
        browser.set_page_load_timeout(25)
        browser.get('http://www.baidu.com')
        browser.find_element_by_name('wd').send_keys(key, Keys.ENTER)

        i = 1
        global page_num
        try:
            while i <= int(page_num):
                print('正在爬取第 %d 页' % i, "\n")
                print("-" * 100)
                i += 1
                time.sleep(3)
                # 获取下一页按钮
                page = browser.find_elements_by_class_name('page-inner_2jZi2')[-1]
                results = browser.find_elements_by_class_name('result')
                # results = browser.find_elements_by_class_name('result c-container new-pmd')
                print(results)

                for result in results:
                    # 获取百度跳转网址
                    url = result.find_element_by_tag_name('a').get_attribute('href')
                    # 获取网址title
                    title = result.find_element_by_tag_name('a').text
                    print(title)
                    my_url = get_url(url)
                    if my_url is False:
                        continue
                    else:
                        browser.close()
                        return my_url  # 返回需要的网址

                page.click()
                print("下一页：")
                continue

        except Exception as e:
            print(e)
    except Exception as e:
        print(e)

    process_stop_time = time.time()
    diff_time = process_stop_time - process_start_time
    # 将计算出来的时间戳转换为结构化时间
    struct_time = time.gmtime(diff_time)
    print("爬取百度链接完成 " + "一共耗时{0}分{1}秒".format(struct_time.tm_min, struct_time.tm_sec))
    browser.close()
    return None


def get_url(url):
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, compress',
        'Accept-Language': 'en-us;q=0.5,en;q=0.3',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'
    }
    try:
        # 请求拒绝重定向
        req = requests.get(url, headers=headers, allow_redirects=False)
        req.encoding = 'utf-8'
        # 获取头部Location 中url地址
        print(req.headers['Location'])

        my_url = req.headers['Location']
        print("-" * 100)
        # 信息是否来自汽车之家
        # if "www.autohome.com.cn" in my_url:  # 汽车之家
        # 新浪，太平洋，58汽车
        if "https://db.auto.sina.com.cn/" in my_url or "https://price.pcauto.com.cn/" in my_url or "https://product.58che.com/" in my_url:
            return my_url
        else:
            return False
    except Exception as e:
        pass


def get_url_main(keyword):
    # keyword = 'python 字符串包含'
    page = 1

    print("查询的关键字：" + keyword)
    print("页数：" + str(page))
    print("Usage :page默认无限大")
    key = keyword
    try:
        global page_num
        page_num = page
    except:
        pass

    a = get_baidu(key)
    if a is None:
        print("\n抱歉，没找到~~~~~~~~~~~~~~~~~~")
    else:
        print("\n找到了！！！！！！！！！！！！！！！！")
    print(a)
    print("\n")
    return a


if __name__ == '__main__':
    get_url_main("工地灰层")
