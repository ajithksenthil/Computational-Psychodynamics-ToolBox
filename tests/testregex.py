import re
import ast

text = """```python
[
    {'subject': 'group of wolves', 'action': 'surrounding', 'objects': ['bison'], 'environment': 'snowy field'},
    {'subject': 'wolves', 'action': 'crowding around', 'objects': ['bison'], 'environment': 'snowy field'},
    {'subject': 'wolves', 'action': 'following', 'objects': ['bison'], 'environment': 'snowy field'},
    {'subject': 'wolves', 'action': 'attacking', 'objects': ['bison'], 'environment': 'snowy field'},
    {'subject': 'bison', 'action': 'running', 'objects': ['wolves'], 'environment': 'snowy field'},
    {'subject': 'bison', 'action': 'running', 'objects': {'other bison', 'wolves'}, 'environment': 'snowy field'},
    {'subject': 'bison', 'action': 'charging', 'objects': ['wolves'], 'environment': 'snowy field'},
    {'subject': 'wolves', 'action': 'chasing', 'objects': ['bison'], 'environment': 'snowy field'},
    {'subject': 'wolves', 'action': 'pursuing', 'objects': ['bison'], 'environment': 'snowy field'},
    {'subject': 'wolves', 'action': 'following', 'objects': ['bison'], 'environment': 'snowy field'},
    {'subject': 'wolves', 'action': 'hunting', 'objects': ['bison'], 'environment': 'snowy field'},
    {'subject': 'bison', 'action': 'escaping', 'objects': ['wolves'], 'environment': 'snowy field'},
    {'subject': 'bison', 'action': 'stumbling', 'objects': ['wolves'], 'environment': 'snowy field'},
    {'subject': 'wolves', 'action': 'attacking', 'objects': ['fallen bison'], 'environment': 'snowy field'},
    {'subject': 'wolves', 'action': 'taking down', 'objects': ['fallen bison'], 'environment': 'snowy field'},
    {'subject': 'wolves', 'action': 'biting', 'objects': ['fallen bison'], 'environment': 'snow"""

# Regular expression pattern to match dictionary-like structures
#pattern = r"\{[^{}]*\}"
# Updated regular expression pattern to match more specific dictionary-like structures
# Regular expression pattern to match more specific dictionary-like structures
pattern = r"\{'[^']+'[^{}]+\}"

# Find all matches
matches = re.findall(pattern, text)

# Safely evaluate each match as a Python dictionary
dict_list = [ast.literal_eval(match) for match in matches]

print(dict_list)