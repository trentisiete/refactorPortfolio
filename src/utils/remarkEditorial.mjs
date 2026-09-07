// Normalize presentation without changing the author's words or inline markup.
const plainText = node => node.value ?? (node.children ?? []).map(plainText).join('');

export default function remarkEditorial() {
  return tree => {
    if (!Array.isArray(tree.children)) return;
    const first = tree.children.find(node => !['yaml', 'toml'].includes(node.type));
    // The page already supplies its h1 from frontmatter. Keep a leading deck as prose.
    if (first?.type === 'heading' && (first.depth === 1 || (first.depth === 2 && plainText(first).length > 110))) {
      first.type = 'paragraph';
      delete first.depth;
      first.data = { ...first.data, hProperties: { className: ['article-deck'] } };
    }
    const walk = node => {
      if (node.type === 'paragraph' && node.children?.length === 1 && node.children[0].type === 'strong') {
        node.data = { ...node.data, hProperties: { className: ['paragraph-heading'] } };
      }
      node.children?.forEach(walk);
    };
    walk(tree);
    const index = tree.children.findIndex(node => node.type === 'heading' && /^(abstract|resumen|zusammenfassung)$/i.test(plainText(node).trim()));
    if (index < 0) return;
    const heading = tree.children[index];
    let end = index + 1;
    while (end < tree.children.length) {
      const node = tree.children[end];
      if (node.type === 'heading' && node.depth <= heading.depth) break;
      end++;
    }
    const children = tree.children.slice(index, end);
    tree.children.splice(index, end - index, {
      type: 'blockquote', children,
      data: { hName: 'section', hProperties: { className: ['article-abstract'], 'aria-label': plainText(heading) } },
    });
  };
}
