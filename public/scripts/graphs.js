/*
 * Gráficas nativas para artículos en markdown.
 *
 * Barras horizontales:
 *   <div class="graph-hbar" data-title="..." data-subtitle="..." data-note="..."
 *        data-max="0.9" data-decimals="4">
 *     <div data-label="Linear SVM" data-value="0.7309"></div>
 *   </div>
 *
 * Etapas numeradas:
 *   <div class="graph-flow" data-title="..." data-note="...">
 *     <div data-step="Corpus">Descripción corta de la etapa</div>
 *   </div>
 *
 * Dos gráficas lado a lado: envuélvelas en <div class="graph-duo">...</div>
 */
(function () {
	'use strict';

	function el(tag, className, text) {
		const node = document.createElement(tag);
		if (className) node.className = className;
		if (text) node.textContent = text;
		return node;
	}

	function head(container, dataset) {
		if (!dataset.title && !dataset.subtitle) return;
		const h = el('div', 'graph__head');
		if (dataset.title) h.appendChild(el('p', 'graph__title', dataset.title));
		if (dataset.subtitle) h.appendChild(el('p', 'graph__subtitle', dataset.subtitle));
		container.appendChild(h);
	}

	function note(container, dataset) {
		if (dataset.note) container.appendChild(el('p', 'graph__note', dataset.note));
	}

	function buildHbar(root) {
		const items = Array.from(root.children)
			.map(c => ({ label: c.dataset.label, value: parseFloat(c.dataset.value) }))
			.filter(i => i.label && Number.isFinite(i.value));
		if (items.length === 0) return;

		const max = parseFloat(root.dataset.max) || Math.max(...items.map(i => i.value));
		const decimals = root.dataset.decimals !== undefined ? parseInt(root.dataset.decimals, 10) : NaN;

		root.textContent = '';
		root.setAttribute('role', 'group');
		if (root.dataset.title) root.setAttribute('aria-label', root.dataset.title);
		head(root, root.dataset);

		const rows = el('div', 'graph__rows');
		items.forEach((item, i) => {
			const row = el('div', 'graph__row');
			row.appendChild(el('span', 'graph__label', item.label));
			const track = el('span', 'graph__track');
			const fill = el('span', 'graph__fill');
			fill.style.setProperty('--graph-w', `${(item.value / max) * 100}%`);
			fill.style.transitionDelay = `${i * 80}ms`;
			track.appendChild(fill);
			row.appendChild(track);
			row.appendChild(el('span', 'graph__value',
				Number.isFinite(decimals) ? item.value.toFixed(decimals) : String(item.value)));
			rows.appendChild(row);
		});
		root.appendChild(rows);
		note(root, root.dataset);
		observe(root);
	}

	function buildFlow(root) {
		const steps = Array.from(root.children)
			.map(c => ({ name: c.dataset.step, desc: c.textContent.trim() }))
			.filter(s => s.name);
		if (steps.length === 0) return;

		root.textContent = '';
		root.setAttribute('role', 'group');
		if (root.dataset.title) root.setAttribute('aria-label', root.dataset.title);
		head(root, root.dataset);

		const list = el('ol', 'graph__steps');
		steps.forEach((step, i) => {
			const li = el('li', 'graph__step');
			const stepHead = el('span', 'graph__step-head');
			stepHead.appendChild(el('span', 'graph__step-num', String(i + 1)));
			stepHead.appendChild(el('span', 'graph__step-name', step.name));
			li.appendChild(stepHead);
			if (step.desc) li.appendChild(el('span', 'graph__step-desc', step.desc));
			list.appendChild(li);
		});
		root.appendChild(list);
		note(root, root.dataset);
		observe(root);
	}

	const observer = 'IntersectionObserver' in window
		? new IntersectionObserver(entries => {
			for (const entry of entries) {
				if (entry.isIntersecting) {
					entry.target.classList.add('graph--in');
					observer.unobserve(entry.target);
				}
			}
		}, { threshold: 0.35 })
		: null;

	function observe(root) {
		if (observer) {
			observer.observe(root);
			// Seguro: si el observer no dispara (entornos sin render), animar igualmente
			setTimeout(() => root.classList.add('graph--in'), 3000);
		} else {
			root.classList.add('graph--in');
		}
	}

	function init() {
		document.querySelectorAll('.graph-hbar').forEach(buildHbar);
		document.querySelectorAll('.graph-flow').forEach(buildFlow);
	}

	if (document.readyState === 'loading') {
		document.addEventListener('DOMContentLoaded', init);
	} else {
		init();
	}
})();
