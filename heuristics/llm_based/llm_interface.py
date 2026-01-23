import json
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def call_llm(api_url: str, api_key: str, model: str, prompt: str) -> Optional[str]:
    
    #地址
    #这是HTTP请求头，用来告诉服务器“谁在请求、发的是什么数据”
    headers = {
        "Authorization": f"Bearer {api_key}", #验证api
        "Content-Type": "application/json",   #数据格式json
    }

    ##数据
    #聊天接口的标准请求格式json
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    #把 Python 字典变成可以发送的字节流
    data = json.dumps(payload).encode("utf-8")
    
    #创建一个 HTTP 请求对象，把所有信息封装好
    req = Request(api_url, data=data, headers=headers, method="POST") #url API地址  method-POST一种请求方式
    print("[LLM] request sent")

    try:
        with urlopen(req, timeout=60) as resp: #发送 HTTP 请求，最多等 60 秒
            raw = resp.read().decode("utf-8")  #读取服务器返回的数据（是字节），解码成字符串
            print("[LLM] raw response:", raw[:200])
            obj = json.loads(raw)  #把返回的 JSON 字符串解析成 Python 字典
            return obj["choices"][0]["message"]["content"] #从返回里拿模型输出文本并返回
    
    except HTTPError as e:
        print(f"[LLM] HTTPError: {e.code} {e.reason}")
        return None
    
    except URLError as e:
        print(f"[LLM] URLError: {e.reason}")
        return None
    
    except json.JSONDecodeError as e:
        print(f"[LLM] JSON decode error: {e}")
        return None
    
    except (KeyError, IndexError) as e:
        print(f"[LLM] Response format error: {e}")
        # 如果需要，可以打印 raw 看原始返回
        return None
