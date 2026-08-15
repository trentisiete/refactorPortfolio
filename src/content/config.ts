import { defineCollection, z } from 'astro:content';

const articles = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    topic: z.string().optional(),
    date: z.date().or(z.string()),
    excerpt: z.string().optional(),
    tags: z.array(z.string()).optional().default([]),
    lang: z.enum(['es', 'en', 'de']).default('es'),
    kind: z.enum(['article', 'blog']).default('article'),
    draft: z.boolean().optional().default(false),
    translate: z.boolean().optional().default(true),
    translated: z.boolean().optional().default(false),
    sourceHash: z.string().optional(),
  }),
});

const thoughts = defineCollection({
  type: 'content',
  schema: z.object({
    publishedAt: z.coerce.date(),
    lang: z.literal('en').default('en'),
    draft: z.boolean().optional().default(false),
    image: z.object({
      src: z.string(),
      alt: z.string(),
      caption: z.string().optional(),
    }).optional(),
  }),
});

export const collections = { articles, thoughts };
