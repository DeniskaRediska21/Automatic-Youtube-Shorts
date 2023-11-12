import g4f

def get_gpt_responce(prompt):

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
    return response
