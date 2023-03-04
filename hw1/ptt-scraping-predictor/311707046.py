import requests
from bs4 import BeautifulSoup
import time
from tqdm.auto import tqdm
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
from datetime import datetime

payload = {
    'from':'/bbs/Beauty/index.html',
    'yes':'yes'
}
rs = requests.session()
res = rs.post("https://www.ptt.cc/ask/over18",data = payload)
def load_article(file):
    with open(file,'r',encoding='utf-8') as f:
        all_article = []
        for line in f:
            obj = json.loads(line)
            all_article.append(obj)
    return all_article
def get_article_by_time(all_article, start,end):
    target_time_article = []
    for article in all_article:
        if (int(article['date'])<=end_date) & (int(article['date'])>=start_date):
            target_time_article.append(article)
    return target_time_article
def date_check(start_date,end_date):
    try:
        if (len(start_date) != 4) | (len(end_date) != 4):
            raise ValueError
        else:
            date_obj = datetime.strptime(start_date, '%m%d')
            date_obj = datetime.strptime(end_date, '%m%d')
            if int(start_date) <= int(end_date):
                return True
            else:
                raise ValueError
    except ValueError as error:
        print("Invalid Date Input")
        return False
# 1. crawl : all article & popular article
if sys.argv[1]=='crawl':
    if not len(sys.argv) == 2:
        print("Input Invalid")
        sys.exit()
    def get_first_post_index():
        i=1
        while True:
            content = rs.get(f"https://www.ptt.cc/bbs/Beauty/index{i}.html")
            soup = BeautifulSoup(content.text, 'html.parser')
            articles = soup.select('div.r-ent')
            for article in articles:
                title = article.find("div", class_="title")
                if title.a is None:
                    continue
                url = str(title.a['href'])
                if url == "/bbs/Beauty/M.1640974182.A.7DB.html":
                    return i
            i+=1
    def get_last_post_index(start_index):
        index = start_index
        tmp_date =""
        while True:
            content = rs.get(f"https://www.ptt.cc/bbs/Beauty/index{index}.html")
            soup = BeautifulSoup(content.text, 'html.parser')
            articles = soup.select('div.r-ent')
            for article in articles:
                date = article.select_one("div.date").text.strip()
                if (tmp_date =="")&(date=="1/01"):
                    tmp_date = date
                if (date=="1/01") & (tmp_date =="12/31"):
                    return index
                if tmp_date !="":
                    tmp_date = date
            index += 1

    def get_article(index):
        global  all_article, popular_article, lock
        content = rs.get(f"https://www.ptt.cc/bbs/Beauty/index{index}.html")
        soup = BeautifulSoup(content.text, 'html.parser')
        num_post = len(soup.select('div.r-ent'))
        for i,s in enumerate(soup.select('div.r-ent')): # 20個貼文
            date = s.select_one("div.date").text.strip()
            if (index == start_index) & (date[:2] =="12"):
                continue
            if (index==end_index)&(date =="1/01"):
                continue
            tmp = {}
            # get title and check
            title = s.find("div", class_="title")
            title_=title.text.strip()
            if "公告"in title_:
                continue
            #get date 
            date = s.find("div", class_="date")
            m,day  = date.text.split('/')
            m = m.strip()
            transform_date = m.zfill(2)+day
            #get url
            if title.a is None:
                continue
            sub_url = str(title.a['href'])
            if not sub_url:
                continue
            url = "https://www.ptt.cc/" + sub_url
            tmp ={
                "date":transform_date,
                "title": title_,
                "url":str(url),
            }
            with lock :
                all_article.append(tmp)
                if s.select_one("div.nrec").text == "爆":
                    popular_article.append(tmp)
        delay_choices = [1,2,3] 
        delay = random.choice(delay_choices) 
        time.sleep(delay)
        index +=1
    all_article = []
    popular_article = []
    lock = threading.Lock()
    start_index = get_first_post_index()
    end_index = get_last_post_index(start_index)

    with ThreadPoolExecutor(max_workers= 10) as executor:
        for i in range(start_index,end_index+1):
            executor.submit(get_article,i)
    jsonl_all_article = ""
    all_article =sorted(all_article, key=lambda x: int(x['date']))
    for article in all_article:
        json_str = json.dumps(article,ensure_ascii=False)
        jsonl_all_article += json_str + "\n"
    jsonl_popular_article = ""
    popular_article =sorted(popular_article, key=lambda x: int(x['date']))
    for article in popular_article:
        json_str = json.dumps(article,ensure_ascii=False)
        jsonl_popular_article += json_str + "\n"
    output_file1 = "all_article.jsonl"
    with open(output_file1,'w',encoding='utf-8') as f:
        f.write(jsonl_all_article)
    output_file2 = "popular_article.jsonl"
    with open(output_file2,'w',encoding='utf-8') as f:
        f.write(jsonl_popular_article)
# 2. push : push_{start_date}_{end_date}.json
elif sys.argv[1]=="push":
    if not len(sys.argv) ==4:
        print("Input Invalid")
        sys.exit()

    def like_boo_search(article):
        global all_like,all_boo,user_count_dict, lock
        content = rs.get(article['url'])
        soup = BeautifulSoup(content.text, 'html.parser')
        pushes = soup.select("div.push")
        for push in pushes:
            tag = push.select_one("span.push-tag").text.strip()
            user_id = push.select_one("span.push-userid").text.strip()
            with lock:
                if user_id not in user_count_dict:
                    user_count_dict[user_id] = [0,0]
                if tag=="推":
                    all_like = all_like + 1
                    user_count_dict[user_id][0]  = user_count_dict[user_id][0]+1
                elif tag=="噓":
                    all_boo = all_boo + 1
                    user_count_dict[user_id][1]  = user_count_dict[user_id][1]+1
        delay_choices = [0.2,0.3,0.4,0.5,0.8] 
        delay = random.choice(delay_choices) 
        time.sleep(delay)
    # read all article in
    if date_check(sys.argv[2],sys.argv[3]):
        start_date = int(sys.argv[2])
        end_date = int(sys.argv[3])
    else:
        sys.exit()
    # load in all article
    all_article = load_article("all_article.jsonl")
    # select target time interval article
    target_article = get_article_by_time(all_article, start_date,end_date)
    # get like & boo
    all_like = 0
    all_boo = 0
    user_count_dict = {}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Use the executor to asynchronously execute the parse_html function for each HTML document in the list
        executor.map(like_boo_search,target_article)
    like_sorted_dict = sorted(user_count_dict.items(), key=lambda item: item[1][0], reverse=True)[:10]
    boo_sorted_dict = sorted(user_count_dict.items(), key=lambda item: item[1][1], reverse=True)[:10]
    like_dict ={}
    boo_dict ={}
    for i in range(0,10):
        like_dict[f"like {i+1}"] = {
            "user_id" : like_sorted_dict[i][0],
            "count" : like_sorted_dict[i][1][0]
        }
    for i in range(0,10):
        boo_dict[f"boo {i+1}"] = {
            "user_id" : boo_sorted_dict[i][0],
            "count" : boo_sorted_dict[i][1][1]
        }
    output_dict ={
        "all_like":all_like,
        "all_boo" : all_boo,
        **like_dict,
        **boo_dict
    }
    start_date_output = str(start_date).zfill(4)
    end_date_output = str(end_date).zfill(4)
    output_file = f"push_{start_date_output}_{end_date_output}.json"
    with open(output_file, 'w') as f:
        json.dump(output_dict,f)
# 3. popular : popular_{start_date}_{end_date}.json
elif sys.argv[1]=="popular":
# read all article in
    if not len(sys.argv) == 4:
        print("Input Invalid")
        sys.exit()
    def popular_search(article):
        global image_url, lock
        content = rs.get(article['url'])
        soup = BeautifulSoup(content.text,"html.parser")
        main = soup.select_one('#main-content')
        for a in main.find_all('a'):
            link = a['href']
            pattern = re.compile(r"^https?://.*\.(jpg|jpeg|png|gif)$", re.IGNORECASE)
            if pattern.match(link):
                with lock:
                    image_url.append(link)
        delay_choices = [0.5,0.8,1,1.3] 
        delay = random.choice(delay_choices) 
        time.sleep(delay)
    image_url = []
    lock = threading.Lock()
    start_date = int(sys.argv[2])
    end_date = int(sys.argv[3])
    popular_article = load_article("popular_article.jsonl")
    target_article = get_article_by_time(popular_article,start_date,end_date)
    number_of_popular_articles = len(target_article)
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Use the executor to asynchronously execute the parse_html function for each HTML document in the list
        executor.map(popular_search, target_article)
    #write file
    start_date_output = str(start_date).zfill(4)
    end_date_output = str(end_date).zfill(4)
    output_file = f"popular_{start_date_output}_{end_date_output}.json"
    with open(output_file,"w") as f:
        json.dump({
        "number_of_popular_articles": number_of_popular_articles,
        "image_urls": image_url
        },f)
# 4. keyword : keyword_{keyword}_{start_date}_{end_date}.json
elif sys.argv[1]=="keyword":
    # read all article in
    if not len(sys.argv) ==5:
        print("Input Invalid")
        sys.exit()
    def keyword_search(article):
        global image_url, lock
        content = rs.get(article['url'])
        soup = BeautifulSoup(content.text,"html.parser")
        f2 = soup.select("span.f2")
        delay_choices = [0.3, 0.4] 
        delay = random.choice(delay_choices) 
        time.sleep(delay)
        flag = 0
        for f in f2:
            if "發信站" in f.text:
                flag = 1
        if flag == 1:
            # 刪除keword搜尋range以外的內容
            main = soup.select_one("div#main-content")
            exclude_tags = main.select('.push,.f2,.article-metaline-right')
            for tag in exclude_tags:
                tag.decompose()
            #判斷keyword是否在article中，若是，則把照片爬下來
            if keyword in main.text.strip():
                soup = BeautifulSoup(content.text,"html.parser")
                new_main = soup.select_one("div#main-content")
                for a in new_main.find_all('a'):
                    link = a['href']
                    pattern = re.compile(r"^https?://.*\.(jpg|jpeg|png|gif)$", re.IGNORECASE)
                    if pattern.match(link):
                        with lock:
                            image_url.append(link)
            delay_choices = [0.3, 0.4, 0.5,0.7, 0.8] 
            delay = random.choice(delay_choices) 
            time.sleep(delay)
    lock = threading.Lock()
    image_url=[]
    start_date = int(sys.argv[3])
    end_date = int(sys.argv[4])
    keyword = str(sys.argv[2])
    all_article = load_article("all_article.jsonl")
    target_article = get_article_by_time(all_article, start_date,end_date)

    with ThreadPoolExecutor(max_workers=10) as executor:
        # Use the executor to asynchronously execute the parse_html function for each HTML document in the list
        executor.map(keyword_search, target_article)
    #write file
    start_date_output = str(start_date).zfill(4)
    end_date_output = str(end_date).zfill(4)
    output_file = f"keyword_{keyword}_{start_date_output}_{end_date_output}.json"
    with open(output_file,"w") as f:
        json.dump({
            "image_urls": image_url
        },f)

else:
    print("First arg is invalid !! Please type in one of the four words including crawl, push, pupular and keyword.")


