import re

file_path = 'src/content/articles/difussion_models.md'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# ============================================================
# 1. Fix math blocks: remove \begin{equation}, \end{equation}, \label{...}
# ============================================================
# Pattern: $$\begin{equation} ... \label{...} ... \end{equation}$$
# We want: $$ ... $$  (without \begin/\end/\label)
content = re.sub(r'\$\$\s*\\begin\{equation\}', '$$', content)
content = re.sub(r'\\end\{equation\}\s*\$\$', '$$', content)
# Remove any remaining \begin{equation} or \end{equation} standalone
content = re.sub(r'\\begin\{equation\}', '', content)
content = re.sub(r'\\end\{equation\}', '', content)
# Remove \label{...} lines inside math blocks
content = re.sub(r'\s*\\label\{[^}]*\}', '', content)

# ============================================================
# 2. Fix citations: [@key] -> [^key] and [^@key] -> [^key]
# ============================================================
# Fix [^@key] (broken from previous partial script)
content = re.sub(r'\[\^@([^\]]+)\]', r'[^\1]', content)
# Fix [@key] (original pandoc format still remaining)
content = re.sub(r'\[@([^\]]+)\]', r'[^\1]', content)

# ============================================================
# 3. Strip pandoc {reference-type="ref" reference="..."} attributes
# ============================================================
# These can span multiple lines, so use DOTALL
# Pattern: [text](#anchor){reference-type="ref" reference="anchor"}
# Result: [text](#anchor)
content = re.sub(
    r'\{reference-type="ref"\s+reference="[^"]*"\}',
    '',
    content,
    flags=re.DOTALL
)

# ============================================================
# 4. Fix escaped quotes \" -> "
# ============================================================
content = content.replace('\\"', '"')

# ============================================================
# 5. Remove data-latex-placement from HTML figures
# ============================================================
content = re.sub(r'\s*data-latex-placement="[^"]*"', '', content)

# ============================================================
# 6. Remove orphan 'center' command (line 988-ish)
# ============================================================
# It's a line that just says " center" by itself
content = re.sub(r'^\s*center\s*$', '', content, flags=re.MULTILINE)

# ============================================================
# 7. Fix escaped brackets \[...\] -> plain text
# ============================================================
content = content.replace('\\[Pylint\\]', 'Pylint')
content = content.replace('\\[MIT\\]', 'MIT')

# ============================================================
# 8. Convert pandoc table (classifier accuracy) to markdown pipe table
# ============================================================
# The pandoc table looks like:
#  {#tab:classifier_accuracy}
#   **Modelo SDE Base del Clasificador**    **Precisión Promedio (Accuracy)**
#   -------------------------------------- -----------------------------------
#   VE-SDE                                                0.55
#   VP-SDE Lineal                                         0.35
#   SubVP-SDE Lineal                                      0.44
#
#   : Precisión promedio de los clasificadores ...

pandoc_table = """ {#tab:classifier_accuracy}
  **Modelo SDE Base del Clasificador**    **Precisión Promedio (Accuracy)**
  -------------------------------------- -----------------------------------
  VE-SDE                                                0.55
  VP-SDE Lineal                                         0.35
  SubVP-SDE Lineal                                      0.44

  : Precisión promedio de los clasificadores 'TimeDependentWideResNet'
  entrenados para diferentes SDEs base sobre CIFAR-10 ruidoso."""

markdown_table = """| Modelo SDE Base del Clasificador | Precisión Promedio (Accuracy) |
|---|---|
| VE-SDE | 0.55 |
| VP-SDE Lineal | 0.35 |
| SubVP-SDE Lineal | 0.44 |

*Tabla: Precisión promedio de los clasificadores TimeDependentWideResNet entrenados para diferentes SDEs base sobre CIFAR-10 ruidoso.*"""

content = content.replace(pandoc_table, markdown_table)

# ============================================================
# 9. Clean up cross-reference equation links like [\[eq:...\]](#eq:...)
# ============================================================
# Pattern: (Ecuación [\[eq:reverse_sde\]](#eq:reverse_sde))
# These have the \[ and \] which are problematic. Let's simplify them.
# Change [\[eq:name\]](#eq:name) -> [Ec.](#eq:name)
content = re.sub(
    r'\[\\?\[eq:([^\]\\]+)\\?\]\]\(#eq:([^\)]+)\)',
    r'[Ec.](#eq:\2)',
    content
)

# ============================================================
# 10. Clean up figure/table cross-ref links that have leftover \[ \]
# ============================================================
# [\[tab:...\]](#tab:...) -> [Tabla](#tab:...)
content = re.sub(
    r'\[\\?\[tab:([^\]\\]+)\\?\]\]\(#tab:([^\)]+)\)',
    r'[Tabla](#tab:\2)',
    content
)
# [\[fig:...\]](#fig:...) -> [Fig.](#fig:...)
content = re.sub(
    r'\[\\?\[fig:([^\]\\]+)\\?\]\]\(#fig:([^\)]+)\)',
    r'[Fig.](#fig:\2)',
    content
)

# ============================================================
# 11. Clean up empty lines left by removals
# ============================================================
content = re.sub(r'\n{4,}', '\n\n\n', content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("All markdown fixes applied successfully.")
