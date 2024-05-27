# byllm
[中文](README.md)

A small LLM evaluation framework.
## Features
- Support batch inference.

## Use
The QA file must be a jsonl file，of which only has two properties: "question",  "right_answer". For example, the [qa_data_ready.jsonl](qa_data_ready.jsonl):
```
{"question": "Princess Alexandrine of Baden (Alexandrine Luise Amalie Friederike Elisabeth Sophie; 6 December 1820 \u2013 20 December 1904) was the Duchess of Saxe-Coburg and Gotha as the wife of Ernest II. She was born the eldest child of Leopold, Grand Duke of Baden and his wife Princess Sophie of Sweden.Ernest II (German: \"Ernst August Karl Johann Leopold Alexander Eduard\"; 21 June 1818 \u2013 22 August 1893) was the sovereign duke of the Duchy of Saxe-Coburg and Gotha, reigning from 1844 to his death. So what country was the grandma of the wife of Ernest II, Duke of Saxe-Coburg and Gotha from?", "right_answer": "Sweden"}
{"question": "Cheryl Stephanie Burke (born May 3, 1984) is an American dancer, model and TV host. She is best known for being one of the professional dancers on ABC's \"Dancing with the Stars\", where she was the first female professional to win the show and the first professional to win twice and consecutively.Dancing with the Stars is an American dance competition television series that premiered on June 1, 2005, on ABC. It is the US version of the UK series \"Strictly Come Dancing\". So Cheryl Stephanie Burke best known for being one of the professional dancers on an American dance competition television series and is the US version of what UK series?", "right_answer": "Strictly Come Dancing"}
```
Write your API key into [api-config.json](api-config.json).

To start an evaluation of Qwen1.5-0.5B-Chat against [qa_data_ready.jsonl](qa_data_ready.jsonl), using judge [deepseek](https://platform.deepseek.com/api_keys),  run
```
python eval_start.py --model /data/Qwen1.5-0.5B-Chat --judge deepseek --qafile /data/byllm/qa_data_ready.jsonl --batchsize 2 --max_new_tokens 100
```
The output is `qa_data_ready_llm_answers_judge.jsonl`。

## Notes
Basic code and notes are in [eval_en.ipynb](eval_en.ipynb).
