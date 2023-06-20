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

## HW2 : model compression
> pruning.ipynb : 紀錄pruning的方法細節

> 以下為 model_training&testing.ipynb的詳細內容

此jupyter notebook主要分為九個部分，依序執行即可得到最後testing accuracy以及model的參數

**1. Basic setting & data loading**
* 將需使用的package import進來
* 設定seed and device
* load in data and transform

**2. Load teacher model**
* 將作業提供的參數讀進來作為teacher model

**3. Build student model**
* 用Depthwise-Separable-Convolution建立student model

**4. Knowledge distillation**
* 用KD將teacher model distill到student model

**5. Testing accuracy for unpruned student_model**
* 在testing data上進行測試，得到student model 在 pruning 前的準確度

**6. Pruning student_model**
* 對student model 進行pruning，使其參數量低於100000

**7. Testing accuracy before retraining**
* 測試pruning後student model在testing data的準確度

**8. Retrain pruned student_model**
* 重新訓練student model

**9. Testing accuracy after retraining pruned student_model**
* 測試不同epoch下retrained student model在testing data的準確度，從中選出最好的作為最終預測用的model參數，並將參數輸出為參數檔

**10. Final testing : Load in final model for testing**
* 將final weights for testing讀入並對testing data進行預測，輸出預測結果

## Hw3 Few shot learning
### Data
* 5 ways 5 shots: 共五個類別，每個類別五張圖片
### Algorithm

**相關套件:**
[learn2learn](https://github.com/learnables/learn2learn/tree/0b9d3a3d540646307ca5debf8ad9c79ffe975e1c)

**1. MAML** : 

分為inner與outer loop，前者用**support set**(5 ways 5 shots)訓練模型，後者用訓練好的模型在query set(5 ways 5 shots)上測試並將loss更新到meta model上

其他參考資源: [How to train your MAML](https://arxiv.org/abs/1810.09502)

**2. Prototypical network** :

將support set經過backbone encoder轉換為embedding後，計算各類別的平均embedding得到prototype，在以query set的embedding，計算各類別到其prototype的距離(cosine similarity/Euclidean distance)並做為loss更新

**3. R2D2**

### 其他技巧
[Data Augmentation for Meta-Learning](https://arxiv.org/abs/2010.07092)

[code](https://github.com/RenkunNi/MetaAug/blob/79d1a6a457be37258df50a9194946caeb86845a2/metadatas/taskaug.py)

1. Task augmentation : 增加原training dataset的類別數，e.g 對圖片做augmentation(rotation)產生新的類別

   (Feature augmentation是直接對類別內的image做augmentation)

3. Query augmentation:對query做augmentation,e.g. Cutmix、Selfmix

## Hw5 headline generation
### Data
a list of **(title,body)** dictionary
### Method
huggingface -BART(效果最好)、T5

resource : 
* summarization : https://huggingface.co/learn/nlp-course/chapter7/5?fw=pt
* summarization : https://huggingface.co/docs/transformers/tasks/summarization#inference
* BART : https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration
* LORA : https://huggingface.co/blog/peft

## Hw6 Node anomaly detection
### Data
**torch_geometric data format**

https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
### Method
* GCN
* GAT
* augmentation
### Resource 
* node classfication : https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing
* pytorch_geometric example : https://github.com/pyg-team/pytorch_geometric/tree/master/examples
* Data format : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
* GNN intro(distill) : https://distill.pub/2021/gnn-intro/
* graph augmentation lib : https://github.com/rish-16/grafog
