def deepseek_factory(api_key, max_new_tokens, base_url):
    from openai import OpenAI
    from tqdm import tqdm

    client = OpenAI(api_key=api_key, base_url=base_url)

    def llm_response(prompts: list[str]) -> list[str]:
        # get a string, return a answer string
        results = []
        for s in prompts:
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": s},
                    ],
                    max_tokens=max_new_tokens,
                    temperature=0.7,
                    stream=False,
                )
                results += [response.choices[0].message.content]
            except:
                results += ["Network Error!"]
        return results

    return llm_response


def baidu_factory(
    api_key,
    secret_key,
    max_new_tokens,
    base_url,
):
    import requests
    import json

    def get_access_token():

        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={secret_key}&client_secret={api_key}"

        payload = json.dumps("")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")

    url = base_url + get_access_token()

    headers = {"Content-Type": "application/json"}

    def llm_response(prompts: list[str]) -> list[str]:
        results = []
        for s in prompts:
            payload = json.dumps({"messages": [{"role": "user", "content": s}], "max_output_tokens": max_new_tokens})
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                results += [response.json()["result"]]
            except:
                results += ["Network Error!"]

        return results

    return llm_response


def hf_factory(model_path, max_new_tokens):

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )

    def llm_response(prompts: list[str]) -> list[str]:
        texts = []
        for prompt in prompts:
            messages = [
                # 有的模型可以省略system prompt
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)


        model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(
            model.device
        )

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return responses

    return llm_response
