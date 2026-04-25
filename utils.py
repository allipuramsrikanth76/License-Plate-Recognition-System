import re

def clean_text(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    return text