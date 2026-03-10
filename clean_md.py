import re

file_path = 'src/content/articles/difussion_models.md'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace block math definitions properly for Rehype/Remark
content = re.sub(r'\$\$\\begin\{equation\}', '$$', content)
content = re.sub(r'\\end\{equation\}\$\$', '$$', content)

# Fix equations labeled with equation environment inside regular display math
content = re.sub(r'\\begin\{equation\}', '', content)
content = re.sub(r'\\end\{equation\}', '', content)

# Fix captions/references, e.g., [#cap:estado_del_arte]{reference-type...} -> [#cap:estado_del_arte]
content = re.sub(r'\[#(.*?)\]\{reference-type="ref"\s*reference=".*?"\}', r'[#\1]', content)

# Fix references to sections with text in front, e.g., Capítulo [#cap]{reference-...}
# We already replaced the reference-type part, so it's now just `[#cap:...]`
# These will render as plain text anchors if link matching fails, but it's simpler.

# Remove `data-latex-placement="H"` from figure tags since Astro/HTML doesn't need them
content = re.sub(r' data-latex-placement=".*?"', '', content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Markdown processing complete.")
