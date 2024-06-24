
def get_gpt_responce(prompt,model = 'mistral'):

    
    if model == 'g4f':
        import g4f

        try:
            provider=g4f.Provider.DeepAi
            response = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
        except:
            try:
                provider=g4f.Provider.Aichat
                response = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
            except:
                try:
                    response = g4f.ChatCompletion.create(model=g4f.Model.gpt_4,provider=g4f.Provider.ChatgptAi	, messages=[{"role": "user", "content": prompt}]) 
                except:
                    try:
                        provider=g4f.Provider.ChatgptLogin
                        response = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
                    except:
                        try:
                            provider=g4f.Provider.AiService
                            response = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
                        except:
                            try:
                                provider=g4f.Provider.Lockchat
                                response = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}])
                            except:
                                provider = g4f.Provider.AItianhu
                                response = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}])

    else:
        from multiprocessing import Process, Manager
        import os
        import Supports.config

        m = Manager()
        q = m.Queue()
        p = Process(target = ollama_responce, args = (prompt,model,q)).start()
        response = q.get()
        #response = ollama_responce(prompt, model)
        command = 'kill -SIGUSR1 $(pgrep -f ollama)'
        os.system('echo %s|sudo -S %s' % (Supports.config.sudopass, command))
        #os.popen("sudo -S %s"%(command), 'w').write(Supports.config.sudopass)

    return response

def ollama_responce(prompt, model, q):


    from langchain_community.llms import Ollama
    import torch
    try:
        with torch.no_grad():
                ollama = Ollama(base_url='http://localhost:11434',
                model=model)
    except: 
        with torch.no_grad():
                ollama = Ollama(base_url='http://localhost:11434',
                model='mistral')

    text_chunk = []

    for chunk in ollama._stream(prompt):
        text_chunk.append(chunk.text)

    response = ''.join(text_chunk)
    q.put(response)
    return response
