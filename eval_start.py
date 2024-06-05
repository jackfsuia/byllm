from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from datasets import load_dataset
import modelsAPI
import os
import argparse

def model_answers(target_model, qa_file, batch_size):

    eval_dataset = load_dataset(os.path.dirname(qa_file), data_files=qa_file, split="train")
    data_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    qa_file_llm_answers = qa_file[:-6] + "_llm_answers.jsonl"

    if target_model.is_hf_model:
        with open(qa_file_llm_answers,'w',encoding='utf-8') as f2:
            for i in tqdm(data_loader, total=len(data_loader)):
                ans = target_model(i['question'])
                for q,a,ra in zip(i['question'], ans, i['right_answer']):
                    my_dict={"question":q,"your_answer":a,"right_answer":ra }
                    f2.write(json.dumps(my_dict)+'\n')
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            with open(qa_file_llm_answers,'w',encoding='utf-8') as f2:
                    for i in tqdm(data_loader, total=len(data_loader)):
                        ans = target_model(i['question'], executor)
                        for q,a,ra in zip(i['question'], ans, i['right_answer']):
                            my_dict={"question":q,"your_answer":a,"right_answer":ra }
                            f2.write(json.dumps(my_dict)+'\n')    
    return qa_file_llm_answers

def model_judges(judge_func, answer_file, jbatchsize):

    eval_dataset = load_dataset(os.path.dirname(answer_file), data_files=answer_file, split="train")
    data_loader = DataLoader(dataset=eval_dataset, batch_size=jbatchsize, shuffle=False, num_workers=8)

    judge_file = answer_file[:-6] + "_judge.jsonl"
    all_q = 0
    right_a = 0
    network_err=0
    template = "Now I give you one question and two answers to it. One of the answers is student's answer, another is the right answer. Please based on the given right answer, judge if the student's answer get\
        it right. If the student get it right, please respond with a 'yes' and reasons, otherwise with a 'no' and reasons.\n Here is the question:{question}.\n \
            Student's answer: {your_answer}. \n Right answer: {right_answer}. "
    if judge_func.is_hf_model:
        with open(judge_file,'w',encoding='utf-8') as f:
            for i in tqdm(data_loader, total=len(data_loader)):
                prompts = []
                for q, ya, ra in zip(i['question'], i['your_answer'], i['right_answer']):


                    prompts += [template.format(question = q, your_answer = ya, right_answer = ra)]


                responses = judge_func(prompts)

                items="" 

                for q, ya, ra, r in zip(i['question'], i['your_answer'], i['right_answer'], responses):

                    label = 0
                    if "yes" in r.lower()[:5]:
                        label = 1
                        right_a += 1
                        all_q += 1
                    elif "Network Error!" == r:
                        label = -100
                        network_err += 1
                    else:
                        all_q += 1
                    item= {"question":q, "your_answer":ya, "right_answer":ra, "label": label, "response": r}
                    items += (json.dumps(item)+'\n')


                f.write(items)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=jbatchsize) as executor:
            with open(judge_file,'w',encoding='utf-8') as f:
                for i in tqdm(data_loader, total=len(data_loader)):
                    prompts = []
                    for q, ya, ra in zip(i['question'], i['your_answer'], i['right_answer']):


                        prompts += [template.format(question = q, your_answer = ya, right_answer = ra)]


                    responses = judge_func(prompts, executor)

                    items="" 

                    for q, ya, ra, r in zip(i['question'], i['your_answer'], i['right_answer'], responses):

                        label = 0
                        if "yes" in r.lower()[:5]:
                            label = 1
                            right_a += 1
                            all_q += 1
                        elif "Network Error!" == r:
                            label = -100
                            network_err += 1
                        else:
                            all_q += 1
                        item= {"question":q, "your_answer":ya, "right_answer":ra, "label": label, "response": r}
                        items += (json.dumps(item)+'\n')


                    f.write(items)
    return right_a, all_q, right_a / all_q, network_err, judge_file

def select_model():

    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument('--model', type=str, default=None,help="model name")
    parser.add_argument('--judge', type=str, default=None, help="judge model")
    parser.add_argument('--qafile', type=str, default="byllm/qa_data_ready.jsonl", help="qa file path")
    parser.add_argument('--afile', type=str, default="byllm/qa_data_ready_llm_answers.jsonl", help="answer file path")
    parser.add_argument('--batchsize', type=int, default=50, help="batch size")
    parser.add_argument('--jbatchsize', type=int, default=50, help="batch size of judge")
    parser.add_argument('--provider', type=str, default=None, help="model from which provider")
    parser.add_argument('--jprovider', type=str, default=None, help="judge from which provider")
    parser.add_argument('--max_new_tokens', type=int, default=512, help="max new tokens generated by model")
    parser.add_argument('--jmax_new_tokens', type=int, default=100, help="max new tokens generated by judge model")


    args = parser.parse_args()

    target_model_config, model_judge_config = {},{}

    with open("api-config.json",'r',encoding="utf-8") as f:
        config = json.load(f)
        for m in config:
            if m['name']== args.model:
                target_model_config = m
            if m['name']== args.judge:
                model_judge_config = m

    if (not target_model_config) and args.model:
        print(f"{args.model} API not found, will be using huggingface model {args.model} as target model. If you still mean API, please add this model in modelsAPI.py, api-config.json, eval_start.py.")
        target_model= modelsAPI.hf_factory(args.model, args.max_new_tokens)
    elif args.model is None:
        print("No target model specified, so will only judge answer.")
        target_model = None
    else:        
        if target_model_config['provider'] == 'baidu':
            target_model= modelsAPI.baidu_factory(target_model_config['api_key'], target_model_config['secret_key'], args.max_new_tokens, target_model_config['base_url'])
        elif target_model_config['provider'] == 'deepseek':
            target_model= modelsAPI.deepseek_factory(target_model_config['api_key'], args.max_new_tokens, target_model_config['base_url'])        
        else:
            raise Exception(f"Provider {target_model_config['provider']} not supported, please add the support in modelsAPI.py and eval_start.py.")    
        
    if (not model_judge_config) and args.judge:
        print(f"{args.judge} API not found, will be using huggingface model {args.judge} as judge. If you still mean API, please add this model in modelsAPI.py, api-config.json, eval_start.py.")
        model_judge= modelsAPI.hf_factory(args.judge, args.jmax_new_tokens)
    elif args.judge is None:
        print("No judge specified, so will only generate answer.")
        model_judge = None
    else:        
        if model_judge_config['provider'] == 'baidu':
            model_judge= modelsAPI.baidu_factory(model_judge_config['api_key'], model_judge_config['secret_key'], args.jmax_new_tokens, model_judge_config['base_url'])
        elif model_judge_config['provider'] == 'deepseek':
            model_judge= modelsAPI.deepseek_factory(model_judge_config['api_key'], args.jmax_new_tokens, model_judge_config['base_url'])        
        else:
            raise Exception(f"Provider {model_judge_config['provider']} not supported, please add the support in modelsAPI.py and eval_start.py.")
    if target_model is None and model_judge is None:
        raise Exception("No target model or judge model!")
    return target_model, model_judge, args

if __name__ == "__main__":

    import time

    # 程序开始运行的时间
    start_time = time.time()
    target_model, model_judge, args = select_model()
    qa_file_llm_answers = args.afile
    if target_model:
        qa_file_llm_answers = model_answers(target_model, args.qafile, args.batchsize)
        end_time = time.time()
    if model_judge:
        right_a, all_q, accuracy, network_err, judge_file = model_judges(model_judge, qa_file_llm_answers, args.jbatchsize)
    execution_time = end_time - start_time
    print(f"程序运行时间: {execution_time}秒")
    print(f"right answers ={right_a}, successful judges = {all_q}, accuracy ={accuracy}. network errors = {network_err}. Results are saved to {judge_file}")

