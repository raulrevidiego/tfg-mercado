// Formato de números y fechas en español. Funciones puras (se prueban con `node --test ui/app/tests`).

/** Número con coma decimal. `–` si no hay valor. */
export function es(x, decimals = 2) {
  if (x == null || Number.isNaN(x)) return '–';
  return x.toFixed(decimals).replace('.', ',').replace(/^-(?=\d)/, '−');
}

/** Fracción a porcentaje: 0.0387 → "3,87 %". */
export function pct(x, decimals = 2) {
  return x == null || Number.isNaN(x) ? '–' : `${es(x * 100, decimals)} %`;
}

/** Con signo explícito: +0,49 % / −0,22 %. */
export function signedPct(x, decimals = 2) {
  if (x == null || Number.isNaN(x)) return '–';
  return `${x >= 0 ? '+' : '−'}${es(Math.abs(x * 100), decimals)} %`;
}

/** p-valor para leer: "< 0,00001" si es el mínimo posible; el resto con 4 decimales. */
export function pValue(p, draws) {
  if (p == null) return '–';
  if (draws && p <= 1 / (draws + 1) + 1e-12) return `< ${es(1 / draws, 5)}`;
  return es(p, 4);
}

export function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

const NY = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' });

/** Sesión de Nueva York (YYYY-MM-DD) de una marca de tiempo en segundos (las velas diarias de Alpaca abren a medianoche de NY). */
export function nyDate(epochSeconds) {
  return NY.format(new Date(epochSeconds * 1000));
}

/** "2026-11-25" → "25 nov 2026". */
export function dateEs(iso) {
  if (!iso) return '–';
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString('es-ES', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
}
