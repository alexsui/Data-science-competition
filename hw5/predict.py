from transformers import AutoTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import torch
import json
import numpy as np
import random
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False
device = "cuda" if torch.cuda.is_available() else "cpu"

#請在此處設定模型參數與測試資料路徑
model_path="version2_bart_30epoch/checkpoint-290000/"
test_data_path = "data/test.json"

model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
res = []

with open(test_data_path,"r") as f:
    data = f.readlines()
    for d in tqdm(data[:-1]):
        try:
            d = eval(d)
            inputs = tokenizer(d['body'], return_tensors="pt",truncation=True).input_ids
            outputs  = model.generate(inputs, max_new_tokens=30, do_sample=False)
            outputs=tokenizer.decode(outputs[0], skip_special_tokens=True)
            res.append({"title":outputs})
        except Exception as e:
            res.append({"title":None})

with open("311707046.json", "w") as file:
     for item in res:
        json.dump(item, file)
        file.write('\n')