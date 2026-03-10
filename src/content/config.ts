import { defineCollection, z } from 'astro:content';

const articles = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.date().or(z.string()),
    excerpt: z.string().optional(),
    tags: z.array(z.string()).optional().default([]),
    lang: z.enum(['es', 'en', 'de']).default('es'),
    readingTime: z.number().optional(),
    draft: z.boolean().optional().default(false),
  }),
});

export const collections = { articles };
