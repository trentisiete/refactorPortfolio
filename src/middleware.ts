import { sequence, defineMiddleware } from 'astro:middleware';
import { middleware } from 'astro:i18n';

const i18nMiddleware = middleware({
    prefixDefaultLocale: true,
    redirectToDefaultLocale: false,
    fallbackType: 'rewrite',
});

export const onRequest = i18nMiddleware;
