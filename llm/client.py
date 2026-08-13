from __future__ import annotations
import json
import urllib.error
import urllib.request

class LLMClientError(RuntimeError):
    pass

class OllamaClient:

    def __init__(self, base_url='http://localhost:11434', model='qwen2.5:7b', temperature=0.3, timeout=120):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt, system=None, temperature=None, max_tokens=1500):
        payload = {'model': self.model, 'prompt': prompt, 'stream': False, 'options': {'temperature': self.temperature if temperature is None else temperature, 'num_predict': max_tokens}}
        if system:
            payload['system'] = system
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(self.base_url + '/api/generate', data=data, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode('utf-8'))
        except urllib.error.URLError as e:
            raise LLMClientError(f'Ollama 连接失败（{self.base_url}）：{e}。请确认已安装并运行 ollama serve，且已 pull {self.model}') from e
        if 'response' not in out:
            raise LLMClientError(f'Ollama 返回异常: {out}')
        return out['response'].strip()

class OpenAIClient:

    def __init__(self, base_url, model, api_key='', temperature=0.3, timeout=120):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt, system=None, temperature=None, max_tokens=1500):
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        payload = {'model': self.model, 'messages': messages, 'temperature': self.temperature if temperature is None else temperature, 'max_tokens': max_tokens}
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        req = urllib.request.Request(self.base_url + '/chat/completions', data=json.dumps(payload).encode('utf-8'), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')[:500]
            raise LLMClientError(f'API 返回 {e.code}: {body}') from e
        except urllib.error.URLError as e:
            raise LLMClientError(f'API 连接失败（{self.base_url}）：{e}') from e
        try:
            return out['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise LLMClientError(f'API 返回格式异常: {str(out)[:300]}') from e

def build_llm_client(cfg):
    llm_cfg = cfg.get('LLM', {})
    provider = (llm_cfg.get('provider') or 'none').lower()
    if provider in ('none', '', 'off'):
        return None
    if provider == 'ollama':
        return OllamaClient(base_url=llm_cfg.get('ollama_base_url', 'http://localhost:11434'), model=llm_cfg.get('ollama_model', 'qwen2.5:7b'), temperature=llm_cfg.get('temperature', 0.3), timeout=int(llm_cfg.get('timeout', 120)))
    if provider == 'openai_compatible':
        base = llm_cfg.get('api_base_url', '')
        if not base:
            raise LLMClientError('provider=openai_compatible 需要配置 LLM.api_base_url')
        return OpenAIClient(base_url=base, model=llm_cfg.get('api_model', 'deepseek-chat'), api_key=llm_cfg.get('api_key', ''), temperature=llm_cfg.get('temperature', 0.3), timeout=int(llm_cfg.get('timeout', 120)))
    raise LLMClientError(f'未知 LLM provider: {provider}（可选 ollama / openai_compatible / none）')
