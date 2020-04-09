import re

def word_slice(text):
    if len(re.split(r'\-{5,}', text))>2:
        text = re.split(r'\-{5,}', text)[2]
    text = re.split(r'底本：', text)[0]
    text = re.sub(r'《.+?》', '', text)
    text = re.sub(r'［＃.+?］', '', text)
    text = re.sub(r'\u3000', '', text)
    text = re.sub(r'\r\n', '', text)
    text = text.strip()
    
    return text
