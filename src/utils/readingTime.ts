import type { Lang } from '../i18n/utils';

const WORDS_PER_MINUTE: Record<Lang, number> = {
  es: 210,
  en: 225,
  de: 200,
};

const countMatches = (text: string, pattern: RegExp) => text.match(pattern)?.length ?? 0;

/**
 * Estimates the real reading time of an article from its Markdown body.
 * Prose uses a language-specific reading speed; code, formulas and visual
 * elements add a small fixed cost instead of inflating the word count.
 */
export function estimateReadingTime(markdown: string, lang: Lang): number {
  let content = markdown.normalize('NFC');
  let extraSeconds = 0;

  content = content.replace(/```[^\n]*\n([\s\S]*?)```/g, (_block, code: string) => {
    const lines = code.trim().split(/\r?\n/).filter(Boolean).length;
    extraSeconds += Math.min(60, Math.max(8, lines * 2));
    return ' ';
  });

  content = content.replace(/\$\$[\s\S]*?\$\$/g, () => {
    extraSeconds += 12;
    return ' ';
  });

  const inlineFormulaCount = countMatches(content, /(?<!\$)\$(?!\$)[^\n$]+\$(?!\$)/g);
  extraSeconds += inlineFormulaCount * 2;
  content = content.replace(/(?<!\$)\$(?!\$)[^\n$]+\$(?!\$)/g, ' ');

  const markdownImageCount = countMatches(content, /!\[[^\]]*\]\([^)]*\)/g);
  const htmlVisualCount = countMatches(content, /<(?:img|canvas)\b[^>]*>/gi);
  extraSeconds += (markdownImageCount + htmlVisualCount) * 8;
  content = content.replace(/!\[[^\]]*\]\([^)]*\)/g, ' ');

  // Keep link labels, but discard their destinations and non-visible markup.
  content = content
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/https?:\/\/\S+/g, ' ')
    .replace(/[`*_>#|~\[\]()-]/g, ' ');

  const words = content.match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu)?.length ?? 0;
  const proseSeconds = (words / WORDS_PER_MINUTE[lang]) * 60;

  return Math.max(1, Math.ceil((proseSeconds + extraSeconds) / 60));
}
