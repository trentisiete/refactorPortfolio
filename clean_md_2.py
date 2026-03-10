import re

file_path = 'src/content/articles/difussion_models.md'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix multi-line reference-type attributes
content = re.sub(r'\[#(.*?)\]\{reference-type="ref"\s+reference=".*?"\}', r'[#\1]', content, flags=re.DOTALL)

# Let's fix citations like [@author2023] -> [author2023]
content = re.sub(r'\[@(.*?)\]', r'[\1]', content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Markdown processing complete.")
