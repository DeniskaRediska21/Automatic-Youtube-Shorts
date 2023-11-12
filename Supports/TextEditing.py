
# class TextEditing:
# def __init__(self) -> None:
    
    
import re 
import numpy as np
# @staticmethod
def split_into_sentences(text: str) -> list[str]:
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|edu|me)"
    digits = "([0-9])"
    multiple_dots = r'\.{2,}'
    
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

# @staticmethod
def cleenup(response):
        response = re.sub("\[.*?\]","",response)
        response = re.sub("\"","",response)
        response = re.sub("Topic:","",response)
        response = ".\n".join([ll.rstrip() for ll in response.splitlines() if ll.strip()])
        response = re.sub("\.\.",".",response)
        response = re.sub("Title: ","",response)
        response = re.sub("Voiceover: ","",response)


        # Надо разделить примерно по 20 слов для генерации изображений 
        responce_split = split_into_sentences(response)
        res_=[]
        for i,sentence in enumerate(responce_split):
            responce_split[i]= re.sub("\.","",sentence)
            if responce_split[i]!= '':
                res_.append(responce_split[i])
        responce_split=res_
            
        print(responce_split)
        # Дальше генерация изображений
        Title = responce_split[0]
        Title = re.sub("\.","",Title)

        Title = re.sub("Title:","",Title)
        Title_=Title

        del(responce_split[responce_split=='.'])
        del(responce_split[0])
        
        Title = re.sub(":","_",Title)
        Title = re.sub(" ","_",Title)
        print(np.size(response.split()))

        return response, responce_split, Title_, Title
    