import re
import os
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def find_filetype(folder:str, filetype:str) -> list[str]:
    return [os.path.join(folder,file)
            for file in sorted_alphanumeric(os.listdir(folder))
            if file.endswith(filetype)]