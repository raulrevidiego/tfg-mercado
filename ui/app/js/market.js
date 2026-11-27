// Pantalla 1: Mercado y eventos. Velas + volumen, eventos 8-K sobre las velas, panel de rango real y previsto
// (HAR-E) y lista de seguimiento. Todo en un único gráfico con dos paneles que comparten el eje de tiempo.
import { getJSON, getJSONorNull } from './api.js';
import { COLORS } from './theme.js';
import { CATEGORY_INFO, DEFAULT_CATEGORIES, buildMarkers, categoryLabel, eventTipHTML, eventsByDate, symbolBanner } from './events.js';
import { dateIndex, nextCardHTML, rangeSeries, reactionPoints } from './range.js';
import { dateEs, es, escapeHtml, nyDate, pct } from './format.js';

const MINUTES = { '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440 };
const RANGES = [['1M', 30], ['3M', 91], ['6M', 182], ['1A', 365], ['5A', 1826], ['Todo', null]];
const BENCH = { source: 'binance_spot', symbol: 'BTCUSDT', interval: '1m', tf: '1h' };
const PREFERRED = 'NVDA';  // donde los saltos en resultados se ven mejor

export function createMarket({ els, onBanner }) {
  const L = window.LightweightCharts;
  const state = {
    symbol: null, kind: 'stock', candles: [], index: null, events: null, vol: null,
    summary: null, meta: null, cats: new Set(DEFAULT_CATEGORIES), layers: { events: true, range: true },
    range: 'Todo', loadId: 0,
  };

  // ---------------------------------------------------------------- gráfico
  const chart = L.createChart(els.chart, {
    autoSize: true,
    layout: { background: { color: COLORS.card }, textColor: COLORS.ink, panes: { separatorColor: COLORS.border } },
    grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
    rightPriceScale: { borderColor: COLORS.border },
    timeScale: { borderColor: COLORS.border, timeVisible: false, secondsVisible: false },
  });
  const candleSeries = chart.addSeries(L.CandlestickSeries, {
    upColor: COLORS.blue, borderUpColor: COLORS.blue, wickUpColor: COLORS.blue,
    downColor: COLORS.card, borderDownColor: COLORS.magenta, wickDownColor: COLORS.magenta, borderVisible: true,
  });
  const volumeSeries = chart.addSeries(L.HistogramSeries, { priceFormat: { type: 'volume' }, priceScaleId: 'vol', lastValueVisible: false });
  chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
  const eventMarkers = L.createSeriesMarkers(candleSeries, []);
  let range = null;  // {real, har, nn, markers}: series del panel de rango (existen solo si la capa está activa)

  // ---------------------------------------------------------------- capas
  function updateEventMarkers() {
    const on = state.layers.events && state.events && state.index;
    eventMarkers.setMarkers(on ? buildMarkers(state.events.events, state.cats, state.index.dateToTime) : []);
  }

  function updateRangeLayer() {
    const want = state.layers.range && state.vol && state.index;
    if (!want) {
      if (range) {
        for (const s of [range.real, range.har]) chart.removeSeries(s);
        range = null;
      }
      return;
    }
    if (!range) {
      const priceFormat = { type: 'custom', formatter: v => pct(v, 1), minMove: 0.0001 };
      const real = chart.addSeries(L.LineSeries, { color: COLORS.ink, lineWidth: 1, priceFormat, priceLineVisible: false, lastValueVisible: false, title: 'real' }, 1);
      const har = chart.addSeries(L.LineSeries, { color: COLORS.blue, lineWidth: 2, priceFormat, priceLineVisible: false, lastValueVisible: true, title: 'HAR-E' }, 1);
      range = { real, har, markers: L.createSeriesMarkers(real, []) };
      chart.panes()[1]?.setHeight(170);
    }
    const { dateToTime } = state.index;
    range.real.setData(rangeSeries(state.vol, 'real', dateToTime));
    range.har.setData(rangeSeries(state.vol, 'har_enext', dateToTime));
    range.markers.setMarkers(reactionPoints(state.vol, dateToTime).map(p => ({
      time: p.time, position: 'inBar', shape: 'circle', color: COLORS.magenta, size: 1,
    })));
  }

  // ---------------------------------------------------------------- tooltip de eventos
  chart.subscribeCrosshairMove(param => {
    const tip = els.tip;
    const date = param.time != null && state.index ? state.index.timeToDate.get(param.time) : null;
    const ev = date && state.layers.events && state.events ? eventsByDate(state.events.events, state.cats).get(date) : null;
    if (!ev || !param.point) { tip.hidden = true; return; }
    tip.innerHTML = eventTipHTML(ev, state.symbol);
    tip.hidden = false;
    const width = els.chart.clientWidth;
    tip.style.left = `${Math.max(4, Math.min(param.point.x + 14, width - tip.offsetWidth - 4))}px`;
    tip.style.top = `${Math.max(4, param.point.y + 14)}px`;
  });

  // ---------------------------------------------------------------- rango visible
  els.ranges.innerHTML = RANGES.map(([label]) => `<button type="button" class="pill" data-r="${label}" aria-pressed="false">${label}</button>`).join('');
  els.ranges.addEventListener('click', e => {
    if (e.target.dataset.r) { state.range = e.target.dataset.r; applyRange(); }
  });

  function applyRange() {
    for (const b of els.ranges.children) {
      const on = b.dataset.r === state.range;
      b.classList.toggle('active', on);
      b.setAttribute('aria-pressed', String(on));
    }
    const days = RANGES.find(r => r[0] === state.range)[1];
    const c = state.candles;
    if (!c.length) return;
    const first = c[0].time, last = c[c.length - 1].time;
    if (days == null) { chart.timeScale().fitContent(); return; }
    const from = Math.max(first, last - days * 86400);
    if (from >= last) chart.timeScale().fitContent(); else chart.timeScale().setVisibleRange({ from, to: last });
  }

  // ---------------------------------------------------------------- controles de capas
  els.layerEvents.addEventListener('change', () => { state.layers.events = els.layerEvents.checked; updateEventMarkers(); });
  els.layerRange.addEventListener('change', () => { state.layers.range = els.layerRange.checked; updateRangeLayer(); applyRange(); });
  els.cats.addEventListener('change', e => {
    if (e.target.type !== 'checkbox') return;
    if (e.target.checked) state.cats.add(e.target.value); else state.cats.delete(e.target.value);
    updateEventMarkers();
  });

  function renderCategoryChips() {
    const counts = {};
    for (const e of state.events?.events ?? []) counts[e.category] = (counts[e.category] ?? 0) + 1;
    els.cats.innerHTML = Object.keys(counts).sort((a, b) => Object.keys(CATEGORY_INFO).indexOf(a) - Object.keys(CATEGORY_INFO).indexOf(b))
      .map(c => `<label class="chip"><input type="checkbox" value="${escapeHtml(c)}"${state.cats.has(c) ? ' checked' : ''}>` +
        `<b>${CATEGORY_INFO[c]?.short ?? '·'}</b> ${escapeHtml(categoryLabel(c))} <small>${counts[c]}</small></label>`).join('');
  }

  // ---------------------------------------------------------------- carga de una acción
  async function select(symbol) {
    const my = ++state.loadId;
    const bench = symbol === BENCH.symbol;
    state.symbol = symbol;
    state.kind = bench ? 'bench' : 'stock';
    els.title.textContent = symbol;
    els.status.textContent = 'cargando…';
    markActiveRow(symbol);
    try {
      const url = bench
        ? `/api/candles?source=${BENCH.source}&symbol=${symbol}&interval=${BENCH.interval}&tf=${BENCH.tf}&limit=5000`
        : `/api/candles?source=alpaca_stocks&symbol=${symbol}&interval=1d&tf=1d&limit=5000`;
      const [candles, events, vol] = await Promise.all([
        getJSON(url),
        bench ? null : getJSONorNull(`/api/app/events?symbol=${symbol}`),
        bench ? null : getJSONorNull(`/api/app/volatility?symbol=${symbol}`),
      ]);
      if (my !== state.loadId) return;  // el usuario ya eligió otra acción
      state.candles = candles;
      state.events = events;
      state.vol = vol;
      state.index = bench ? null : dateIndex(candles);

      candleSeries.setData(candles.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
      volumeSeries.setData(candles.map(d => ({ time: d.time, value: d.volume, color: (d.close >= d.open ? COLORS.blue : COLORS.magenta) + '55' })));
      chart.applyOptions({ timeScale: { timeVisible: bench } });

      els.layerEvents.disabled = !events;
      els.layerRange.disabled = !vol;
      renderCategoryChips();
      updateEventMarkers();
      updateRangeLayer();
      els.next.innerHTML = vol ? nextCardHTML(vol.next) : '';
      els.next.hidden = !vol?.next;
      applyRange();

      const last = candles[candles.length - 1];
      els.status.textContent = `${candles.length.toLocaleString('es-ES')} velas · último cierre ${es(last.close)} · ${dateEs(nyDate(last.time))}`;
      els.note.textContent = bench
        ? 'Banco de pruebas: tres meses de bitcoin de 2024 a 1 minuto (velas de 1 hora). Sin eventos ni previsión de rango.'
        : !events ? 'Referencia del mercado: sin eventos ni previsión de rango.' : '';
      onBanner(bench ? 'Bitcoin (banco de pruebas): sirvió para probar los datos y el código, no entra en el estudio de eventos.'
        : events ? symbolBanner(symbol, state.summary, events.events) : `${symbol}: referencia del mercado (sin eventos propios).`);
    } catch (err) {
      if (my === state.loadId) els.status.textContent = err.message;
    }
  }

  // ---------------------------------------------------------------- lista de seguimiento
  function markActiveRow(symbol) {
    for (const tr of els.watch.querySelectorAll('tr[data-s]')) tr.classList.toggle('active', tr.dataset.s === symbol);
  }

  const change = v => v == null ? '–' : `<span class="${v >= 0 ? 'up' : 'down'}">${v >= 0 ? '+' : '−'}${es(Math.abs(v))} %</span>`;

  async function loadWatchlist() {
    const [rows, datasets] = await Promise.all([getJSON('/api/watchlist'), getJSON('/api/datasets')]);
    const mine = new Set(state.meta?.symbols ?? rows.map(r => r.symbol));
    const group = (title, list) => list.length
      ? `<tr class="group"><th colspan="4">${title}</th></tr>` + list.map(r =>
        `<tr data-s="${escapeHtml(r.symbol)}" tabindex="0"><td><b>${escapeHtml(r.symbol)}</b></td><td>${es(r.close)}</td><td>${change(r.change_1d)}</td><td>${change(r.change_1m)}</td></tr>`).join('')
      : '';
    const hasBench = datasets.some(d => d.symbol === BENCH.symbol && d.source === BENCH.source);
    els.watch.innerHTML = `<table><thead><tr><th>Acción</th><th>Cierre</th><th>1D</th><th>1M</th></tr></thead><tbody>` +
      group('Acciones del estudio', rows.filter(r => mine.has(r.symbol))) +
      group('Referencias del mercado', rows.filter(r => !mine.has(r.symbol))) +
      (hasBench ? `<tr class="group"><th colspan="4">Banco de pruebas</th></tr><tr data-s="${BENCH.symbol}" tabindex="0"><td><b>${BENCH.symbol}</b></td><td colspan="3" class="muted">1 min · 2024</td></tr>` : '') +
      `</tbody></table>`;
    return rows.filter(r => mine.has(r.symbol)).map(r => r.symbol);
  }

  els.watch.addEventListener('click', e => {
    const tr = e.target.closest('tr[data-s]');
    if (tr) select(tr.dataset.s);
  });
  els.watch.addEventListener('keydown', e => {
    const tr = e.target.closest('tr[data-s]');
    if (tr && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); select(tr.dataset.s); }
  });

  // ---------------------------------------------------------------- arranque
  async function init({ meta, summary }) {
    state.meta = meta;
    state.summary = summary;
    const stocks = await loadWatchlist();
    if (!stocks.length) { els.status.textContent = 'No hay datos: ejecuta el ingestor de Alpaca'; return; }
    await select(stocks.includes(PREFERRED) ? PREFERRED : stocks[0]);
  }

  return { init, select, state, chart };
}
