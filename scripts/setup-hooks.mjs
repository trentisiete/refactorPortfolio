// Activa los hooks versionados del repo (.githooks). Se ejecuta en postinstall.
import { execFileSync } from 'node:child_process';

try {
	execFileSync('git', ['config', 'core.hooksPath', '.githooks']);
	console.log('[hooks] core.hooksPath -> .githooks');
} catch {
	// fuera de un repo git (p. ej. CI sobre un tarball): no pasa nada
}
