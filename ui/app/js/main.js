// Arranque de la interfaz: pestañas, lectura rápida de cada pantalla, motor y ayuda.
import { getJSONorNull, getJSON } from './api.js';
import { dateEs } from './format.js';
import { createMarket } from './market.js';

const $ = id => document.getElementById(id);
const banners = { market: '', models: '', events: '', whatif: 'Versión provisional: el simulador definitivo llega en el paso 25. Describe el pasado, no predice.' };
let activeTab = 'market';

function renderBanner() {
  $('banner').hidden = !banners[activeTab];
  $('banner-text').textContent = banners[activeTab] ?? '';
}

function setBanner(tab, text) {
  banners[tab] = text;
  if (tab === activeTab) renderBanner();
}

function showTab(name) {
  activeTab = name;
  for (const tab of document.querySelectorAll('.tab')) tab.setAttribute('aria-selected', String(tab.dataset.tab === name));
  for (const view of document.querySelectorAll('.view')) view.hidden = view.id !== `view-${name}`;
  renderBanner();
}

function notice(message) {
  $('notice').textContent = message;
  $('notice').hidden = !message;
}

async function start() {
  for (const tab of document.querySelectorAll('.tab')) tab.addEventListener('click', () => showTab(tab.dataset.tab));
  $('help-btn').addEventListener('click', () => $('help').showModal());

  const market = createMarket({
    els: {
      chart: $('chart'), tip: $('event-tip'), title: $('symbol-title'), status: $('status'), ranges: $('ranges'),
      layerEvents: $('layer-events'), layerRange: $('layer-range'), cats: $('event-cats'), next: $('next-card'),
      note: $('chart-note'), watch: $('watch'),
    },
    onBanner: text => setBanner('market', text),
  });

  let meta = null, summary = null, models = null;
  try {
    [meta, summary, models] = await Promise.all([
      getJSONorNull('/api/app/meta'), getJSONorNull('/api/app/events-summary'), getJSONorNull('/api/app/models'),
    ]);
  } catch (err) {
    notice(err.message);
  }
  if (!meta) notice('No hay datos precalculados para la interfaz: ejecuta «python -m backend.app.build» desde la carpeta del proyecto y recarga la página.');
  else $('asof').textContent = `Datos hasta ${dateEs(meta.data_as_of)}`;
  if (summary) { $('events-headline').textContent = summary.headline; setBanner('events', summary.headline); }
  if (models) { $('models-headline').textContent = models.headline; setBanner('models', models.headline); }

  getJSON('/api/app/engine').then(e => {
    $('engine').classList.toggle('cpp', e.cpp);
    $('engine-text').textContent = e.cpp ? `Motor: C++ · ${e.threads} ${e.threads === 1 ? 'hilo' : 'hilos'}` : 'Motor: Python (sin compilar)';
  }).catch(() => { $('engine-text').textContent = 'Motor: ?'; });

  await market.init({ meta, summary });
}

start().catch(err => notice(err.message));
