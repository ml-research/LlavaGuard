from llavaguard.server.other_servers import OpenAIModerationsServer, OpenAIServer

try:
    from llavaguard.server.lmdeploy import lmdeployServer
except ImportError:
    lmdeployServer = None

try:
    from llavaguard.server.sglang import SGLangServer, SGLangServerAsync
    from llavaguard.server.sglang_native import SGLangServerNative, SGLangServerNativeAsync
    from llavaguard.server.sglang_offline import SGLangServerOffline, SGLangServerOfflineAsync
    from llavaguard.server.other_servers import LlamaGuardServer
except ImportError:
    SGLangServer, SGLangServerAsync = None, None
    SGLangServerNative, SGLangServerNativeAsync = None, None
    SGLangServerOffline, SGLangServerOfflineAsync = None, None
    LlamaGuardServer = None
    

try:
    from llavaguard.server.vllm import VLLMServer
except ImportError:
    VLLMServer = None

def set_up_server(engine='auto', model_dir=None, device=0):
    quantized = 'quantized' in engine
    engine = engine.replace('_quantized', '')
    if engine == 'auto':
        if model_dir.split('/')[-1] == 'Llama-Guard-3-11B-Vision':
            server = LlamaGuardServer
        elif 'llava-v1.5' in model_dir or 'Llama-3.2-90B-Vision-Instruct' in model_dir:
            print('Running LlavaGuard evaluation using sequential sglang')
            server = SGLangServer
        elif 'intern' in model_dir.lower():
            server = lmdeployServer
        elif 'o1' in model_dir.lower() or 'gpt-4o' in model_dir.lower() or'o1-mini' in model_dir.lower():
            server = OpenAIServer
        elif 'omni-moderation' in model_dir.lower():
            server = OpenAIModerationsServer
        else:
            server = SGLangServerAsync
    elif engine == 'sglang':
        server = SGLangServer
    elif engine == 'sglang_async':
        server = SGLangServerAsync
    elif engine == 'sglang_native':
        server = SGLangServerNative
    elif engine == 'sglang_native':
        server = SGLangServerNativeAsync
    elif engine == 'sglang_offline':
        server = SGLangServerOffline
    elif engine == 'sglang_offline_async':
        server = SGLangServerOfflineAsync
    elif engine == 'lmdeploy':
        server = lmdeployServer
    elif engine == 'vllm':
        server = VLLMServer
    else:
        raise ValueError(f'Engine {engine} not supported')
    if server is None:
        raise ImportError(f'Please install the appropiate libs for engine {engine}')
    
    return server(model_dir=model_dir, device=device, quantized=quantized)
