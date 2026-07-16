/**
 * Traducción automática de contenido.
 *
 * Artículos: src/content/articles/{es,en,de}/<mismo-nombre>.md
 *   - Un archivo SIN `translated: true` es fuente (escrito a mano).
 *   - Un archivo CON `translated: true` fue generado por este script a partir
 *     de la fuente cuyo hash guarda en `sourceHash`.
 *
 * Trayectoria: src/data/profile.{es,en,de}.ts
 *   - El archivo sin cabecera `// translated: true` es la fuente.
 *   - Los generados llevan `// translated: true` y `// sourceHash: <hash>`
 *     en las primeras líneas.
 *
 * Reglas comunes:
 *   - Solo se traducen fuentes con `draft: false` (o sin draft, en artículos).
 *   - Nunca se sobreescribe un archivo no marcado como generado (manual).
 *   - Si `sourceHash` coincide con el hash actual de la fuente, no se retraduce.
 *
 * Uso:
 *   node scripts/translate.mjs            traduce lo pendiente
 *   node scripts/translate.mjs --check    lista lo pendiente sin llamar a la API
 *   node scripts/translate.mjs --hook     modo pre-commit: traduce, hace git add
 *                                         de lo generado y nunca falla (exit 0)
 */

import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const LANGS = ['es', 'en', 'de'];
const LANG_NAMES = { es: 'Spanish', en: 'English', de: 'German' };
const MODEL = process.env.TRANSLATE_MODEL || 'claude-sonnet-5';
const API_URL = 'https://api.anthropic.com/v1/messages';
const CHUNK_LIMIT = 15000; // chars de fuente por llamada
const MAX_TOKENS = 32000;

const args = process.argv.slice(2);
const CHECK_ONLY = args.includes('--check');
const HOOK_MODE = args.includes('--hook');

function log(msg) {
	console.log(`[translate] ${msg}`);
}

function loadApiKey() {
	if (process.env.ANTHROPIC_API_KEY) return process.env.ANTHROPIC_API_KEY;
	const envPath = join(ROOT, '.env');
	if (!existsSync(envPath)) return null;
	for (const line of readFileSync(envPath, 'utf-8').split(/\r?\n/)) {
		const m = line.match(/^\s*ANTHROPIC_API_KEY\s*=\s*"?([^"#\s]+)"?\s*$/);
		if (m) return m[1];
	}
	return null;
}

function normalize(text) {
	return text.replace(/^﻿/, '').replace(/\r\n/g, '\n');
}

function hashOf(text) {
	return createHash('sha256').update(normalize(text)).digest('hex').slice(0, 16);
}

/* ---------- frontmatter helpers (parsing ligero, sin deps) ---------- */

function splitFrontmatter(md) {
	const m = normalize(md).match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
	if (!m) return null;
	return { frontmatter: m[1], body: m[2] };
}

function fmFlag(frontmatter, key) {
	const re = new RegExp(`^${key}\\s*:\\s*true\\s*$`, 'm');
	return re.test(frontmatter);
}

function stripFmKeys(frontmatter, keys) {
	return frontmatter
		.split('\n')
		.filter(line => !keys.some(k => new RegExp(`^${k}\\s*:`).test(line)))
		.join('\n')
		.replace(/\n+$/, '');
}

function markGenerated(frontmatter, lang, sourceHash) {
	const cleaned = stripFmKeys(frontmatter, ['lang', 'translated', 'sourceHash']);
	return `${cleaned}\nlang: "${lang}"\ntranslated: true\nsourceHash: "${sourceHash}"`;
}

/* ---------- helpers para los .ts de trayectoria ---------- */

function capitalize(lang) {
	return lang[0].toUpperCase() + lang.slice(1);
}

function tsIsGenerated(raw) {
	return /^\/\/ translated: true/m.test(raw.slice(0, 200));
}

function tsSourceHash(raw) {
	return raw.slice(0, 200).match(/^\/\/ sourceHash: ([0-9a-f]+)/m)?.[1];
}

function tsStripMarker(raw) {
	return normalize(raw).replace(/^\/\/ translated: true\n\/\/ sourceHash: [0-9a-f]+\n/, '');
}

function tsMarker(hash) {
	return `// translated: true\n// sourceHash: ${hash}\n`;
}

/* ---------- llamadas a la API ---------- */

async function callClaude(apiKey, system, userText) {
	let lastErr;
	for (let attempt = 1; attempt <= 3; attempt++) {
		try {
			const res = await fetch(API_URL, {
				method: 'POST',
				headers: {
					'x-api-key': apiKey,
					'anthropic-version': '2023-06-01',
					'content-type': 'application/json',
				},
				body: JSON.stringify({
					model: MODEL,
					max_tokens: MAX_TOKENS,
					system,
					messages: [{ role: 'user', content: userText }],
				}),
			});
			if (res.status === 429 || res.status >= 500) {
				lastErr = new Error(`API ${res.status}: ${await res.text()}`);
			} else if (!res.ok) {
				throw new Error(`API ${res.status}: ${await res.text()}`);
			} else {
				const data = await res.json();
				if (data.stop_reason === 'max_tokens') {
					throw new Error('Respuesta truncada (max_tokens); el fragmento es demasiado largo');
				}
				return stripFences(data.content.map(b => b.text || '').join(''));
			}
		} catch (err) {
			if (err.message?.startsWith('API 4') || err.message?.includes('truncada')) throw err;
			lastErr = err;
		}
		await new Promise(r => setTimeout(r, attempt * 4000));
	}
	throw lastErr;
}

function stripFences(text) {
	let t = text.trim();
	const fence = t.match(/^```[a-z]*\n([\s\S]*?)\n```$/);
	if (fence) t = fence[1];
	return t;
}

function translationRules(sourceLang, targetLang) {
	return [
		`You translate content for a personal, minimalist tech blog from ${LANG_NAMES[sourceLang]} to ${LANG_NAMES[targetLang]}.`,
		'Rules:',
		'- Preserve the Markdown/YAML structure EXACTLY: same headings levels, lists, blockquotes, footnotes, horizontal rules.',
		'- NEVER translate or alter: code blocks, inline code, LaTeX/KaTeX math ($...$, $$...$$), URLs, link targets, footnote identifiers, HTML tags.',
		'- Exception inside chart HTML blocks (class="graph-hbar" / "graph-flow"): translate the human-readable VALUES of the data-title, data-subtitle, data-note, data-label and data-step attributes; keep data-value, data-max, data-decimals, class names and the markup structure unchanged.',
		'- Translate link display text, but not the destination.',
		'- In YAML frontmatter: keep every key name untouched; translate only the VALUES of title, topic, excerpt and tags; keep date, draft, readingTime and any other value unchanged, including commented-out lines (translate their values too, keeping the # prefix).',
		'- Keep established technical terms in their conventional form (e.g. machine learning terms commonly left in English stay in English).',
		'- Keep the author\'s tone: sober, personal, first person, reflective.',
		'- Output ONLY the translated content. No commentary, no explanations, no code fences around the whole output.',
	].join('\n');
}

async function translateMarkdownFile(apiKey, raw, sourceLang, targetLang) {
	const system = translationRules(sourceLang, targetLang);
	const text = normalize(raw);

	if (text.length <= CHUNK_LIMIT) {
		const out = await callClaude(apiKey, system, `Translate this complete Markdown file (including its YAML frontmatter):\n\n${text}`);
		if (!out.startsWith('---')) throw new Error('La salida no empieza con frontmatter (---)');
		return out;
	}

	// Archivo largo: frontmatter aparte y cuerpo por bloques de secciones ##
	const parts = splitFrontmatter(text);
	if (!parts) throw new Error('No se encontró frontmatter');

	const fmOut = await callClaude(
		apiKey, system,
		`Translate this YAML frontmatter block (return it with the --- delimiters):\n\n---\n${parts.frontmatter}\n---`
	);
	if (!fmOut.startsWith('---')) throw new Error('Frontmatter traducido inválido');

	// Trocear por secciones ## y, si una sección supera el límite,
	// subdividirla por párrafos (los cortes conservan el texto exacto).
	const pieces = [];
	for (const section of parts.body.split(/(?=\n## )/)) {
		if (section.length <= CHUNK_LIMIT) {
			pieces.push(section);
		} else {
			pieces.push(...section.split(/(?<=\n\n)/));
		}
	}

	const chunks = [];
	let current = '';
	for (const s of pieces) {
		if (current && (current.length + s.length) > CHUNK_LIMIT) {
			chunks.push(current);
			current = s;
		} else {
			current += s;
		}
	}
	if (current) chunks.push(current);

	const translated = [];
	for (let i = 0; i < chunks.length; i++) {
		log(`    fragmento ${i + 1}/${chunks.length}...`);
		const out = await callClaude(
			apiKey, system,
			`This is part ${i + 1} of ${chunks.length} of an article's Markdown body (the frontmatter was translated separately). It may start or end mid-section; translate it as-is without completing or closing anything:\n\n${chunks[i]}`
		);
		translated.push(out.trim());
	}
	return `${fmOut}\n\n${translated.join('\n\n')}\n`;
}

async function translateProfileFile(apiKey, raw, sourceLang, targetLang, previous) {
	const cap = capitalize(targetLang);
	const system = [
		`You translate the data file of a personal, minimalist portfolio from ${LANG_NAMES[sourceLang]} to ${LANG_NAMES[targetLang]}.`,
		'The input is a TypeScript module. Output ONLY the complete translated TypeScript file, no commentary, no code fences.',
		'Rules:',
		'- Preserve the code structure, property names, quote style and indentation exactly.',
		`- Rename the exported constants to the target language suffix: profile${capitalize(sourceLang)} -> profile${cap}, profileProjects${capitalize(sourceLang)} -> profileProjects${cap}.`,
		"- Do NOT redeclare interfaces or types: replace any interface/type declarations with a single type-only import of the used types from './profile.es'. If the input already imports them, keep the import as is.",
		'- Translate ONLY human-visible string values: title, intro, tabsLabel, section labels, job titles, locations, details, link labels, image alt texts, excerpts.',
		'- Keep proper nouns and brand names untouched (Naudit HPCN, DripInLocker, Trainologym, Google Cloud...). Institutional names may use their conventional form in the target language.',
		'- Keep untouched: years, URLs, file paths (/logos/...), ids, booleans. In period strings translate only words ("actualidad" -> "present"/"heute", " y " -> " and "/" und ").',
		`- Internal links that start with "/${sourceLang}/" must be rewritten to "/${targetLang}/".`,
		'- Impeccable spelling with all diacritics of the target language.',
	].join('\n');

	let user = `Translate this TypeScript data file:\n\n${tsStripMarker(raw)}`;
	if (previous) {
		user += `\n\nFor reference only, this was the previous translation of an OLDER version of the file. Reuse its terminology and phrasing for entries that have not changed, but the NEW source above is the single source of truth for structure: every entry, property and link present in the new source must appear in your output (and nothing that is no longer there), even if the reference lacks it:\n\n${tsStripMarker(previous)}`;
	}

	const out = await callClaude(apiKey, system, user);
	if (!out.includes(`profile${cap}`) || !out.includes(`profileProjects${cap}`)) {
		throw new Error(`La salida no exporta profile${cap}/profileProjects${cap}`);
	}
	if (!out.includes("./profile.es") && targetLang !== 'es') {
		throw new Error('La salida no importa los tipos de ./profile.es');
	}
	return out;
}

/* ---------- detección de trabajo pendiente ---------- */

function listFiles(dir, ext) {
	if (!existsSync(dir)) return [];
	return readdirSync(dir).filter(f => f.endsWith(ext)).sort();
}

function collectPending() {
	const pending = [];

	for (const lang of LANGS) {
		const dir = join(ROOT, 'src', 'content', 'articles', lang);
		for (const name of listFiles(dir, '.md')) {
			const path = join(dir, name);
			const raw = readFileSync(path, 'utf-8');
			const parts = splitFrontmatter(raw);
			if (!parts) { log(`AVISO: ${lang}/${name} sin frontmatter, ignorado`); continue; }

			const isGenerated = fmFlag(parts.frontmatter, 'translated');
			const isDraft = fmFlag(parts.frontmatter, 'draft');
			if (isGenerated || isDraft) continue;

			const hash = hashOf(raw);
			for (const target of LANGS.filter(l => l !== lang)) {
				const targetPath = join(ROOT, 'src', 'content', 'articles', target, name);
				if (existsSync(targetPath)) {
					const tRaw = readFileSync(targetPath, 'utf-8');
					const tParts = splitFrontmatter(tRaw);
					const tGenerated = tParts ? fmFlag(tParts.frontmatter, 'translated') : false;
					const tHash = tParts?.frontmatter.match(/^sourceHash\s*:\s*"?([0-9a-f]+)"?\s*$/m)?.[1];
					if (!tGenerated) continue;          // traducción manual: no tocar
					if (tHash === hash) continue;        // al día
				}
				pending.push({ kind: 'article', name, sourceLang: lang, targetLang: target, sourcePath: path, targetPath, hash });
			}
		}
	}

	// Trayectoria: profile.<lang>.ts
	for (const lang of LANGS) {
		const path = join(ROOT, 'src', 'data', `profile.${lang}.ts`);
		if (!existsSync(path)) continue;
		const raw = readFileSync(path, 'utf-8');
		if (tsIsGenerated(raw)) continue;

		const hash = hashOf(raw);
		for (const target of LANGS.filter(l => l !== lang)) {
			const targetPath = join(ROOT, 'src', 'data', `profile.${target}.ts`);
			if (existsSync(targetPath)) {
				const tRaw = readFileSync(targetPath, 'utf-8');
				if (!tsIsGenerated(tRaw)) continue;      // traducción manual: no tocar
				if (tsSourceHash(tRaw) === hash) continue; // al día
			}
			pending.push({ kind: 'profile', name: `profile.${target}.ts`, sourceLang: lang, targetLang: target, sourcePath: path, targetPath, hash });
		}
	}
	return pending;
}

/* ---------- main ---------- */

async function main() {
	const pending = collectPending();

	if (pending.length === 0) {
		if (!HOOK_MODE) log('Todo al día: no hay nada que traducir.');
		return 0;
	}

	log(`Pendiente de traducir (${pending.length}):`);
	for (const p of pending) {
		log(`  ${p.sourceLang}/${p.name} -> ${p.targetLang}`);
	}
	if (CHECK_ONLY) return 0;

	const apiKey = loadApiKey();
	if (!apiKey) {
		log('AVISO: no hay ANTHROPIC_API_KEY (ni en el entorno ni en .env). No se traduce nada.');
		return HOOK_MODE ? 0 : 1;
	}

	const written = [];
	let failures = 0;
	for (const p of pending) {
		log(`Traduciendo ${p.sourceLang}/${p.name} -> ${p.targetLang} ...`);
		try {
			const raw = readFileSync(p.sourcePath, 'utf-8');
			let output;
			if (p.kind === 'profile') {
				const previous = existsSync(p.targetPath) ? readFileSync(p.targetPath, 'utf-8') : null;
				const ts = await translateProfileFile(apiKey, raw, p.sourceLang, p.targetLang, previous);
				output = tsMarker(p.hash) + ts.replace(/\n*$/, '\n');
			} else {
				const md = await translateMarkdownFile(apiKey, raw, p.sourceLang, p.targetLang);
				const parts = splitFrontmatter(md);
				if (!parts) throw new Error('La traducción no tiene frontmatter válido');
				output = `---\n${markGenerated(parts.frontmatter, p.targetLang, p.hash)}\n---\n${parts.body.replace(/\n*$/, '\n')}`;
			}
			mkdirSync(dirname(p.targetPath), { recursive: true });
			writeFileSync(p.targetPath, output, 'utf-8');
			written.push(p.targetPath);
			log(`  OK -> ${p.targetLang}/${p.name}`);
		} catch (err) {
			failures++;
			log(`  ERROR en ${p.name} -> ${p.targetLang}: ${err.message}`);
		}
	}

	if (HOOK_MODE && written.length > 0) {
		try {
			execFileSync('git', ['add', ...written], { cwd: ROOT });
			log(`Añadidas al commit ${written.length} traducciones.`);
		} catch (err) {
			log(`AVISO: no se pudo hacer git add: ${err.message}`);
		}
	}

	if (failures > 0) {
		log(`Terminado con ${failures} errores (se reintentará en el próximo commit).`);
		return HOOK_MODE ? 0 : 1;
	}
	log(`Terminado: ${written.length} archivos generados.`);
	return 0;
}

main().then(
	code => process.exit(code),
	err => {
		log(`ERROR inesperado: ${err.message}`);
		process.exit(HOOK_MODE ? 0 : 1);
	}
);
