# Data-science-project
## HW1 : A project for ptt scraping and bulid predictor
### 任務一:
> related file : scrape_ptt.py

**爬PTT Beauty板，並設計四種功能**
* **Crawl** : 爬2022一整年的文章，又分為all_article & popular article(爆文)
```python==
python scrape_ptt.py crawl
```
* **Popular** : 計算日期內爆文的數量並抓取日期內爆文的所有圖片URL
```python==
python scrape_ptt.py popular start_date end_date
```
* **Push** : 計算給訂日期區間內的推文和噓文的數量以及找出日期內最會推跟最會噓的人前10名
```python==
python scrape_ptt.py push start_date end_date
```
* **keyword** : 透過輸入keyword以及日期區間過濾日期內的文章
```python==
python scrape_ptt.py keyword {keyword} start_date end_date
```
### 任務二:
> related file : image_processing.ipynb -> model_training.ipynb -> 311707046_pred.py

**訓練一個binary predictor預測圖片是否為popular(是否屬於推文數>35的ptt文章)**
* **方法一 : 兩分類訓練**
1. 從ppt將照片連結爬下來，若來源文章的推文數大於35，則label為popular image；反之，則label為not popular image
2. 將連結轉換為圖檔，並確定圖檔是否可用PIL package的Image.open()打開 (pytorch 在讀圖檔時使用此function)
3. 從圖中crop 出人臉，以減少雜訊
4. 以4:1切出訓練與測試資料集
5. 透過finetune Resnet，訓練Binary classfication

* **方法二 : Regression訓練** 
1. 從ppt將照片連結爬下來，以來源文章的推文數作為分數，做為爆文則為100，負分則為0
2. 將連結轉換為圖檔，並確定圖檔是否可用PIL package的Image.open()打開 (**pytorch 在讀圖檔時使用此function**)
3. 從圖中crop 出人臉，以減少雜訊
4. 以4:1切出訓練與測試資料集
5. Finetune Resnet，訓練regression model，預測圖片分數
6. 將預測的分數做轉換，若分數>=35，則為popular；反之，則為not popular

* **方法三 : 兩分類訓練(version 2)** 
1. 從ppt將照片連結爬下來，以來源文章的推文數作為分數，**分數高於50的照片為popular，分數低於3的照片為not popular**(**為了平衡正負data的數量**)
2. 將連結轉換為圖檔，並確定圖檔是否可用PIL package的Image.open()打開 (pytorch 在讀圖檔時使用此function)
3. 從圖中crop 出人臉，以減少雜訊
4. 以4:1切出訓練與測試資料集
5. Finetune Resnet，訓練Binary classfication。
由於Resnet訓練時是用彩色照片，所以不可用grayscale transform在圖片上。此外，augmentation的方式選擇flip、rotation與jitter較適合。

