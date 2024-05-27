

def deepseek_factory(api_key,base_url='https://api.deepseek.com'):
    from openai import OpenAI
    from tqdm import tqdm
    import json

    client = OpenAI(
        api_key=api_key, base_url=base_url
    )
    def llm_eval(s):
        # get a string, return a answer string
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": s},
            ],
            max_tokens=100,
            temperature=0.7,
            stream=False,
        )
        return response.choices[0].message.content
    return llm_eval


def baidu_factory(api_key, secret_key, base_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed?access_token="):
    import requests
    import json
    def get_access_token():
    
            
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={secret_key}&client_secret={api_key}"
        
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")
    
    
    
    url = base_url + get_access_token()
    
    headers = {
        'Content-Type': 'application/json'
    }

    def llm_eval(s):

        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": s
                }
            ]
        })

        
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json()['result']
    
    return llm_eval

def hf_factory(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

    def llm_eval(s):

        model_inputs = tokenizer([s], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    return llm_eval


