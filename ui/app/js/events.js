// Eventos (8-K) sobre el gráfico: marcadores, tooltips y la «lectura rápida» de cada acción. Funciones puras.
import { dateEs, escapeHtml, es, pct, signedPct } from './format.js';

export const CATEGORY_INFO = {
  resultados: { label: 'Resultados', short: 'R' },
  regfd: { label: 'Divulgación a inversores', short: 'F' },
  directivos: { label: 'Directivos', short: 'D' },
  otros_eventos: { label: 'Otros eventos', short: 'O' },
  otros: { label: 'Otros', short: '·' },
};
export const DEFAULT_CATEGORIES = ['resultados'];

export function categoryLabel(category) {
  return CATEGORY_INFO[category]?.label ?? category;
}

/** Marcadores de Lightweight Charts: un cuadrado gris oscuro con la inicial de la categoría, sobre la vela. */
export function buildMarkers(events, enabled, dateToTime, color = '#1E242B') {
  const on = new Set(enabled);
  return events
    .filter(e => on.has(e.category) && dateToTime.has(e.session))
    .map(e => ({
      time: dateToTime.get(e.session), position: 'aboveBar', shape: 'square', color,
      text: CATEGORY_INFO[e.category]?.short ?? '·', size: 1,
    }))
    .sort((a, b) => a.time - b.time);
}

/** Mapa fecha → evento (solo categorías activas). Los eventos de una acción son independientes: como mucho uno por día. */
export function eventsByDate(events, enabled) {
  const on = new Set(enabled);
  return new Map(events.filter(e => on.has(e.category)).map(e => [e.session, e]));
}

export function eventTipHTML(ev, symbol) {
  const dir = ev.car1 == null ? '' : ev.car1 >= 0 ? 'sube' : 'baja';
  const rows = [
    ['Reacción (día del evento + siguiente)', `<b>${signedPct(ev.car1)}</b>${dir ? ` · ${dir} más que el mercado` : ''}`],
    ['Solo el día del evento', signedPct(ev.car0)],
    ['Sorpresa (z)', ev.z1 == null ? '–' : `${ev.z1 >= 0 ? '+' : '−'}${es(Math.abs(ev.z1), 1)}`],
    ['Rango real / previsto', ev.abn_range == null ? '–' : `${es(ev.abn_range, 2)}×`],
  ];
  return `<div class="tip-title">${dateEs(ev.session)} · ${escapeHtml(symbol)} · ${escapeHtml(categoryLabel(ev.category))}</div>` +
    rows.map(([k, v]) => `<div class="tip-row"><span>${k}</span><span>${v}</span></div>`).join('');
}

/** «Lectura rápida» de la pantalla de mercado para una acción, escrita desde los números del estudio. */
export function symbolBanner(symbol, summary, events) {
  const results = (events ?? []).filter(e => e.category === 'resultados');
  const cat = summary?.categories?.find(c => c.category === 'resultados');
  const row = cat?.robustness?.by_stock?.find(s => s.symbol === symbol);
  if (row) {
    let text = `${symbol}: los resultados trimestrales mueven la acción un ${pct(row.mean_abs)} de media (2 días, sin signo) frente a un ` +
      `${pct(row.expected_abs)} de un día normal (×${es(row.ratio, 1)}; ${row.n} avisos).`;
    if (cat.robustness.exceptions?.includes(symbol)) text += ' Es una excepción: aquí no se distinguen de un día normal.';
    return text;
  }
  if (results.length) {
    const mean = results.reduce((s, e) => s + Math.abs(e.car1), 0) / results.length;
    return `${symbol}: ${results.length} avisos de resultados; la acción se movió de media un ${pct(mean)} (2 días, sin signo).`;
  }
  return `${symbol}: sin eventos de resultados en los datos.`;
}
