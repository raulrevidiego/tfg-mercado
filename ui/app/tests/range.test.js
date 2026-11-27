import test from 'node:test';
import assert from 'node:assert/strict';
import { dateIndex, nextCardHTML, rangeSeries, reactionPoints } from '../js/range.js';

const CANDLES = [
  { time: Date.UTC(2026, 10, 23, 5) / 1000 }, { time: Date.UTC(2026, 10, 24, 5) / 1000 }, { time: Date.UTC(2026, 10, 25, 5) / 1000 },
];
const VOL = {
  sessions: ['2026-11-23', '2026-11-24', '2026-11-25', '2026-11-27'],
  real: [0.02, 0.03, 0.05, 0.01], har_enext: [null, 0.025, 0.03, 0.02], nn: null, type: [0, 1, 2, 0],
};

test('dateIndex enlaza fechas de sesión y marcas de tiempo de las velas', () => {
  const { dateToTime, timeToDate } = dateIndex(CANDLES);
  assert.equal(dateToTime.get('2026-11-24'), CANDLES[1].time);
  assert.equal(timeToDate.get(CANDLES[2].time), '2026-11-25');
});

test('rangeSeries omite huecos y fechas sin vela, y va ordenada', () => {
  const { dateToTime } = dateIndex(CANDLES);
  assert.deepEqual(rangeSeries(VOL, 'har_enext', dateToTime).map(p => p.value), [0.025, 0.03]);
  assert.equal(rangeSeries(VOL, 'real', dateToTime).length, 3);   // 2026-11-27 no tiene vela
  assert.deepEqual(rangeSeries(VOL, 'nn', dateToTime), []);       // sin predicciones de la red
});

test('reactionPoints marca solo los días de reacción a resultados (tipo 1)', () => {
  const { dateToTime } = dateIndex(CANDLES);
  assert.deepEqual(reactionPoints(VOL, dateToTime), [{ time: CANDLES[1].time, value: 0.03 }]);
});

test('nextCardHTML muestra los dos escenarios y la razón entre ellos', () => {
  const html = nextCardHTML({ as_of: '2026-11-25', har_e_no_event: 0.0192, har_e_event: 0.0431 });
  assert.match(html, /datos hasta 25 nov 2026/);
  assert.match(html, /<b>1,92 %<\/b>/);
  assert.match(html, /<b>4,31 %<\/b>/);
  assert.match(html, /×2,2/);
  assert.equal(nextCardHTML(null), '');
});
