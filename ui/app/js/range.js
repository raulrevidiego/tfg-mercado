// Rango diario real y previsto (HAR-E, red neuronal) y rango esperado de la siguiente sesión. Funciones puras.
import { dateEs, nyDate, pct } from './format.js';

/** Mapas fecha ↔ marca de tiempo de las velas, para colocar eventos y rangos exactamente sobre cada vela. */
export function dateIndex(candles) {
  const dateToTime = new Map();
  const timeToDate = new Map();
  for (const c of candles) {
    const d = nyDate(c.time);
    dateToTime.set(d, c.time);
    timeToDate.set(c.time, d);
  }
  return { dateToTime, timeToDate };
}

/** Serie {time, value} de una columna de volatility.json, sin los huecos (antes del calentamiento no hay predicción). */
export function rangeSeries(vol, key, dateToTime) {
  const values = vol[key];
  if (!values) return [];
  const out = [];
  vol.sessions.forEach((d, i) => {
    const t = dateToTime.get(d);
    if (t != null && values[i] != null) out.push({ time: t, value: values[i] });
  });
  return out.sort((a, b) => a.time - b.time);
}

/** Puntos del rango real en los días de reacción a resultados (tipo 1), para marcarlos. */
export function reactionPoints(vol, dateToTime) {
  const out = [];
  vol.sessions.forEach((d, i) => {
    const t = dateToTime.get(d);
    if (t != null && vol.type[i] === 1) out.push({ time: t, value: vol.real[i] });
  });
  return out.sort((a, b) => a.time - b.time);
}

/** Tarjeta «rango esperado de la siguiente sesión» (HAR-E, con y sin resultados mañana). */
export function nextCardHTML(next) {
  if (!next) return '';
  const ratio = next.har_e_no_event > 0 ? next.har_e_event / next.har_e_no_event : null;
  return `<span class="next-title">Rango esperado de la siguiente sesión <small>(datos hasta ${dateEs(next.as_of)})</small></span>` +
    `<span class="next-val"><small>sin resultados</small> <b>${pct(next.har_e_no_event)}</b></span>` +
    `<span class="next-val"><small>con resultados mañana</small> <b>${pct(next.har_e_event)}</b>` +
    `${ratio ? ` <small>(×${ratio.toFixed(1).replace('.', ',')})</small>` : ''}</span>`;
}
