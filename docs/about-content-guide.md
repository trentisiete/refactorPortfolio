# Guía para añadir contenido a About me

Esta guía explica cómo añadir pensamientos, blogs y música al área
`About me` sin cambiar su diseño.

## Regla principal: About me solo existe en inglés

- Escribe todo el contenido visible en inglés.
- Usa siempre `lang: en`.
- No crees carpetas `/es/about/` ni `/de/about/`.
- Los blogs personales deben llevar `translate: false`.
- No edites nunca la carpeta `dist/`: se vuelve a generar en cada compilación.

## Ver los cambios en local

Desde la raíz del proyecto:

```powershell
npm.cmd run dev
```

Después abre:

- About: <http://127.0.0.1:4321/en/about/>
- Writing: <http://127.0.0.1:4321/en/about/writing/>
- Music: <http://127.0.0.1:4321/en/about/music/>

Antes de publicar, comprueba que todo compila:

```powershell
npm.cmd run build
```

---

## 1. Añadir un pensamiento

Los pensamientos viven en:

```text
src/content/thoughts/
```

### Paso 1

Copia [`src/content/thoughts/template.md`](../src/content/thoughts/template.md)
y ponle un nombre descriptivo, en minúsculas y sin espacios:

```text
2026-07-25-learning-in-public.md
```

El nombre sirve para organizarte, pero no controla la fecha mostrada.

### Paso 2

Usa esta estructura:

```markdown
---
publishedAt: 2026-07-25T20:30:00+02:00
lang: en
draft: true
---

Today I realised that unfinished ideas are often more useful than polished
answers.
```

- `publishedAt` contiene fecha y hora.
- En horario de verano de Madrid usa `+02:00`.
- En horario de invierno usa `+01:00`.
- `draft: true` mantiene el pensamiento oculto.
- Cambia a `draft: false` cuando quieras publicarlo.

Puedes usar Markdown en el texto:

```markdown
**bold**
*italic*
[a link](https://example.com)
```

Los pensamientos se ordenan automáticamente del más nuevo al más antiguo. El
día, la fecha y la hora se calculan y aparecen sin que tengas que escribirlos
en el texto.

---

## 2. Escribir y publicar un blog

Los blogs ingleses viven en:

```text
src/content/articles/en/
```

### Paso 1

Crea un archivo con un nombre corto que pueda funcionar como URL:

```text
src/content/articles/en/why-i-keep-notes.md
```

Su dirección será:

```text
/en/articles/why-i-keep-notes/
```

### Paso 2

Empieza el archivo con este bloque:

```markdown
---
title: "Why I keep notes"
date: 2026-07-25
excerpt: "A short description that will appear in the Writing archive."
tags: ["notes", "learning"]
lang: "en"
kind: "blog"
draft: true
translate: false
---

## The first section

Write the article here.

## Another section

Continue writing in Markdown.
```

Campos importantes:

| Campo | Función |
|---|---|
| `title` | Título del blog |
| `date` | Fecha de publicación y orden en el archivo |
| `excerpt` | Resumen corto mostrado en Writing |
| `tags` | Etiquetas opcionales |
| `lang: "en"` | Hace que pertenezca al About inglés |
| `kind: "blog"` | Hace que aparezca en Writing |
| `draft: true` | Lo mantiene oculto mientras escribes |
| `translate: false` | Evita crear versiones española y alemana |

Cuando esté listo, cambia:

```yaml
draft: false
```

El tiempo de lectura se calcula automáticamente. No tienes que escribirlo.

### Markdown útil para blogs

```markdown
## Heading
### Smaller heading

**Bold text** and *italic text*.

> A quotation.

- One item
- Another item

1. First step
2. Second step

[Link text](https://example.com)

![Image description](/assets/my-image.png)
```

Para usar una imagen, guárdala dentro de `public/assets/` y enlázala empezando
por `/assets/`.

Para ecuaciones, tablas, notas al pie y artículos técnicos consulta también
[`docs/writing-guide.md`](writing-guide.md).

---

## 3. Añadir música de YouTube o Spotify

Las playlists y los álbumes se gestionan desde un único archivo:

[`src/data/aboutPlaylists.ts`](../src/data/aboutPlaylists.ts)

Cada elemento tiene cinco campos:

```ts
{
  platform: 'youtube',
  kind: 'playlist',
  id: 'IDENTIFICADOR',
  title: 'Title shown on the card.',
  description: 'One short description in English.',
},
```

Para añadir uno nuevo, copia uno de los bloques existentes y pégalo dentro de
la lista `aboutPlaylists`.

### Playlist de YouTube

Si el enlace es:

```text
https://youtube.com/playlist?list=PLyi4gdcJFtgeRCwYNR_KtqdzUYSUUSYDk&si=...
```

El identificador es lo que aparece después de `list=` y antes de `&`:

```text
PLyi4gdcJFtgeRCwYNR_KtqdzUYSUUSYDk
```

Añádelo así:

```ts
{
  platform: 'youtube',
  kind: 'playlist',
  id: 'PLyi4gdcJFtgeRCwYNR_KtqdzUYSUUSYDk',
  title: 'A playlist to watch.',
  description: 'Videos and songs gathered in one slightly unruly queue.',
},
```

### Playlist de Spotify

Si el enlace es:

```text
https://open.spotify.com/playlist/40NMQCa8DmUnHMKmvwq1fg?si=...
```

El identificador está después de `/playlist/` y antes de `?`:

```text
40NMQCa8DmUnHMKmvwq1fg
```

Añádelo así:

```ts
{
  platform: 'spotify',
  kind: 'playlist',
  id: '40NMQCa8DmUnHMKmvwq1fg',
  title: 'Things on repeat.',
  description: 'A playlist for whatever has been following me around lately.',
},
```

### Álbum de Spotify

Si el enlace es:

```text
https://open.spotify.com/album/1weenld61qoidwYuZ1GESA?si=...
```

El identificador está después de `/album/` y antes de `?`:

```text
1weenld61qoidwYuZ1GESA
```

Añádelo usando `kind: 'album'`:

```ts
{
  platform: 'spotify',
  kind: 'album',
  id: '1weenld61qoidwYuZ1GESA',
  title: 'My first vinyl.',
  description: 'My girlfriend Lorena gave me this jazz masterpiece on vinyl.',
},
```

Reglas:

- `platform` solo puede ser `'youtube'` o `'spotify'`.
- `kind` puede ser `'playlist'` o `'album'`.
- En `id` pega únicamente el identificador, no el enlace completo.
- Escribe `title` y `description` en inglés.
- El orden de los bloques determina el orden en la página.
- La numeración y el total de selecciones se actualizan automáticamente.
- El reproductor y el enlace externo se generan automáticamente.
- La playlist o el álbum debe ser público o accesible mediante enlace.

---

## Checklist rápido

- [ ] Todo el contenido del About está en inglés.
- [ ] Los pensamientos publicados tienen `draft: false`.
- [ ] Los blogs tienen `lang: "en"` y `kind: "blog"`.
- [ ] Los blogs personales tienen `translate: false`.
- [ ] El identificador de cada playlist o álbum no contiene `?`, `&` ni la URL completa.
- [ ] La web funciona con `npm.cmd run build`.
