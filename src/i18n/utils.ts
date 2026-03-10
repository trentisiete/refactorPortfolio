import es from './es.json';
import en from './en.json';
import de from './de.json';

type Translations = typeof es;

const translations: Record<string, Translations> = { es, en, de };

export const languages = {
    es: 'Español',
    en: 'English',
    de: 'Deutsch',
} as const;

export const defaultLang = 'es';

export type Lang = keyof typeof languages;

export function getLangFromUrl(url: URL): Lang {
    const [, lang] = url.pathname.split('/');
    if (lang in languages) return lang as Lang;
    return defaultLang;
}

export function useTranslations(lang: Lang): Translations {
    return translations[lang] ?? translations[defaultLang];
}

export function getLocalizedPath(path: string, lang: Lang): string {
    // Remove any existing lang prefix
    const cleanPath = path.replace(/^\/(es|en|de)/, '');
    return `/${lang}${cleanPath || '/'}`;
}
