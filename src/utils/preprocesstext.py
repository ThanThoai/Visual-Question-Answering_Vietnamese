import re

punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']

comma_strip = re.compile("(\d)(\,)(\d)")

period_strip = re.compile("(?!<=\d)(\.)(?!\d)")

manumal_map = {
    'khong' : '0',
    'mot' : '1',
    'hai' : '2',
    'ba' : '3',
    'bon' : '4',
    'nam' : '5',
    'sau' : '6',
    'bay' : '7',
    'tam' : '8',
    'chin' : '9',
    'muoi' : '10'
}

def process_punctuation(text):
    out_text = text 
    for p in punct:
        if (p + ' ' in text or ' ' + p in text) or (re.search(comma_strip, text) != None):
            out_text = out_text.replace(p, '')
        else:
            out_text = out_text.replace(p, ' ')
    out_text = period_strip.sub("", out_text, re.UNICODE)
    return out_text

def process_digit(text):
    out_text = []
    temp_text = text.lower().split()
    for word in temp_text:
        word = manumal_map.setdefault(word, word)
        out_text.append(word)
    return ' '.join(out_text)

def prep_ans(answer):
    answer = process_digit(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer
