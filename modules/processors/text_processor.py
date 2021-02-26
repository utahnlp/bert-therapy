import re
from functools import lru_cache

def lru_encode(tokenize_func):
  @lru_cache(maxsize=20)
  def _closure(sentence):
    return tokenize_func.__call__(filter_text(sentence), add_special_tokens=False)
  return _closure

@lru_cache(maxsize=20, typed=True)
def filter_text(text):
    text = re.sub(r"(\[\d*:*\d*\])", "", text)
    return text

