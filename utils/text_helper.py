import re
import numpy as np


def clean_text(rgx_list, text):
  new_text = text
  for rgx_match in rgx_list:
    new_text = re.sub(rgx_match, '', new_text)
  return new_text

def remove_doc_class(c):
  start_idx, end_idx = 0,0 
  if '\documentclass' in c and '\end{document}' in c:
    start_idx = c.find('\documentclass')
    end_idx = c.rfind('\end{document}')
    c = c[:start_idx] + c[end_idx+len('\end{document}'):]
  length = end_idx - start_idx
  return c, length

def unicode_perc(string):
  ascii_array = np.array([s.isascii() for s in string]).astype(int)
  if len(ascii_array) == 0:
    return 0
  return sum(ascii_array) / len(ascii_array)
