# byllm
[English](README_EN.md)

一个简易的大模型评测框架
## 特性
- 支持批量评测。
## 使用
输入的问答数据集必须是一个jsonl文件，满足每行数据只有"question"和"right_answer"两个属性。比如文件[qa_data_ready.jsonl](qa_data_ready.jsonl):
```
{"question": "Princess Alexandrine of Baden (Alexandrine Luise Amalie Friederike Elisabeth Sophie; 6 December 1820 \u2013 20 December 1904) was the Duchess of Saxe-Coburg and Gotha as the wife of Ernest II. She was born the eldest child of Leopold, Grand Duke of Baden and his wife Princess Sophie of Sweden.Ernest II (German: \"Ernst August Karl Johann Leopold Alexander Eduard\"; 21 June 1818 \u2013 22 August 1893) was the sovereign duke of the Duchy of Saxe-Coburg and Gotha, reigning from 1844 to his death. So what country was the grandma of the wife of Ernest II, Duke of Saxe-Coburg and Gotha from?", "right_answer": "Sweden"}
{"question": "Cheryl Stephanie Burke (born May 3, 1984) is an American dancer, model and TV host. She is best known for being one of the professional dancers on ABC's \"Dancing with the Stars\", where she was the first female professional to win the show and the first professional to win twice and consecutively.Dancing with the Stars is an American dance competition television series that premiered on June 1, 2005, on ABC. It is the US version of the UK series \"Strictly Come Dancing\". So Cheryl Stephanie Burke best known for being one of the professional dancers on an American dance competition television series and is the US version of what UK series?", "right_answer": "Strictly Come Dancing"}
```
然后把你的API key写进[api-config.json](api-config.json)。

假如对 Qwen1.5-0.5B-Chat 做评测，用的问答数据集是[qa_data_ready.jsonl](qa_data_ready.jsonl), 用的评测大模型是[deepseek](https://platform.deepseek.com/api_keys), 那么开始评测，只要运行
```
python eval_start.py --model /data/Qwen1.5-0.5B-Chat --judge deepseek --qafile /data/byllm/qa_data_ready.jsonl --batchsize 50 --max_new_tokens 512 --jbatchsize 10 --jmax_new_tokens 100
```
或者互换他俩的角色，那么就是运行
```
python eval_start.py --model deepseek --judge /data/Qwen1.5-0.5B-Chat --qafile /data/byllm/qa_data_ready.jsonl --batchsize 50 --max_new_tokens 512 --jbatchsize 10 --jmax_new_tokens 100
```
即可。评测结果在`qa_data_ready_llm_answers_judge.jsonl`。
## 说明
基本代码和详细的说明都在[eval.ipynb](eval.ipynb)文件。
