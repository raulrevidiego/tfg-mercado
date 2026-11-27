import test from 'node:test';
import assert from 'node:assert/strict';
import { dateEs, es, escapeHtml, nyDate, pValue, pct, signedPct } from '../js/format.js';

test('es: coma decimal, signo menos y valores vacíos', () => {
  assert.equal(es(3.8712), '3,87');
  assert.equal(es(-14.806, 2), '−14,81');
  assert.equal(es(2.35, 1), '2,4');   // el redondeo de JS (2.35 → 2.4 con toFixed) coincide con el del resto del proyecto
  assert.equal(es(null), '–');
  assert.equal(es(Number.NaN), '–');
});

test('pct y signedPct pasan de fracción a porcentaje', () => {
  assert.equal(pct(0.0387), '3,87 %');
  assert.equal(pct(0.01234, 1), '1,2 %');
  assert.equal(signedPct(0.0049), '+0,49 %');
  assert.equal(signedPct(-0.0022), '−0,22 %');
  assert.equal(pct(undefined), '–');
});

test('pValue muestra el p mínimo como cota', () => {
  assert.equal(pValue(1 / 200001, 200000), '< 0,00001');
  assert.equal(pValue(0.0293, 200000), '0,0293');
  assert.equal(pValue(null, 200000), '–');
});

test('escapeHtml evita inyectar HTML', () => {
  assert.equal(escapeHtml('<b>"A&B"</b>'), '&lt;b&gt;&quot;A&amp;B&quot;&lt;/b&gt;');
  assert.equal(escapeHtml(null), '');
});

test('nyDate: la vela diaria de medianoche de Nueva York cae en su sesión, con horario de verano y de invierno', () => {
  assert.equal(nyDate(Date.UTC(2026, 6, 15, 4) / 1000), '2026-07-15');   // verano: medianoche NY = 04:00 UTC
  assert.equal(nyDate(Date.UTC(2026, 10, 25, 5) / 1000), '2026-11-25');  // invierno: 05:00 UTC
  assert.equal(nyDate(Date.UTC(2026, 10, 25, 0) / 1000), '2026-11-24');  // 00:00 UTC aún es el día anterior en Nueva York
});

test('dateEs', () => {
  assert.equal(dateEs('2026-11-25'), '25 nov 2026');
  assert.equal(dateEs(null), '–');
});
