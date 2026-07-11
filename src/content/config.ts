import { defineCollection, z } from 'astro:content';

const articles = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.date().or(z.string()),
    excerpt: z.string().optional(),
    tags: z.array(z.string()).optional().default([]),
    lang: z.enum(['es', 'en', 'de']).default('es'),
    kind: z.enum(['article', 'blog']).default('article'),
    readingTime: z.number().optional(),
    draft: z.boolean().optional().default(false),
    translated: z.boolean().optional().default(false),
    sourceHash: z.string().optional(),
  }),
});

export const collections = { articles };
