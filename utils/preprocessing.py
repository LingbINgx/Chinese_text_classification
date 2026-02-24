import re

def clean_text(text: str):
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[\t\r]', '', text)
    text = text.lower()
    text = re.sub(r' +', ' ', text)

    return text.strip()


def encode_text(text: str, tokenizer, max_length: int = 256):
    text = clean_text(text)
    return tokenizer(
        text,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors="pt",
    )