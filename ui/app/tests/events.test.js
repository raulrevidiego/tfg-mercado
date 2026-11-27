import test from 'node:test';
import assert from 'node:assert/strict';
import { buildMarkers, categoryLabel, eventTipHTML, eventsByDate, symbolBanner } from '../js/events.js';

const EVENTS = [
  { session: '2023-05-25', category: 'resultados', car0: 0.1964, car1: 0.1909, z1: 5.7, abn_range: 2.6 },
  { session: '2023-03-01', category: 'directivos', car0: -0.01, car1: -0.012, z1: -0.5, abn_range: 1.0 },
  { session: '2024-02-22', category: 'resultados', car0: 0.1127, car1: 0.1131, z1: 4.5, abn_range: 1.72 },
];
const DATE_TO_TIME = new Map([['2023-05-25', 1685000000], ['2023-03-01', 1677600000], ['2024-02-22', 1708600000]]);

test('buildMarkers: solo categorías activas, ordenados y con la inicial', () => {
  const m = buildMarkers(EVENTS, ['resultados'], DATE_TO_TIME);
  assert.deepEqual(m.map(x => x.time), [1685000000, 1708600000]);
  assert.ok(m.every(x => x.text === 'R' && x.shape === 'square' && x.position === 'aboveBar'));
  const todos = buildMarkers(EVENTS, ['resultados', 'directivos'], DATE_TO_TIME);
  assert.deepEqual(todos.map(x => x.text), ['D', 'R', 'R']);
  assert.deepEqual(buildMarkers(EVENTS, [], DATE_TO_TIME), []);
});

test('buildMarkers ignora eventos sin vela ese día', () => {
  assert.equal(buildMarkers(EVENTS, ['resultados'], new Map([['2023-05-25', 1]])).length, 1);
});

test('eventsByDate', () => {
  const map = eventsByDate(EVENTS, ['resultados']);
  assert.equal(map.size, 2);
  assert.equal(map.get('2023-05-25').car1, 0.1909);
  assert.equal(map.has('2023-03-01'), false);
});

test('eventTipHTML da la reacción con signo y escapa el símbolo', () => {
  const html = eventTipHTML(EVENTS[0], 'NV<DA>');
  assert.match(html, /25 may 2023 · NV&lt;DA&gt; · Resultados/);
  assert.match(html, /\+19,09 %/);
  assert.match(html, /\+5,7/);
  assert.match(html, /2,60×/);
  assert.equal(categoryLabel('regfd'), 'Divulgación a inversores');
  assert.equal(categoryLabel('nueva'), 'nueva');
});

const SUMMARY = {
  categories: [{
    category: 'resultados',
    robustness: {
      exceptions: ['XOM'],
      by_stock: [
        { symbol: 'NVDA', n: 27, mean_abs: 0.0563, expected_abs: 0.0223, ratio: 2.52 },
        { symbol: 'XOM', n: 27, mean_abs: 0.0181, expected_abs: 0.0181, ratio: 1.0 },
      ],
    },
  }],
};

test('symbolBanner se escribe con los números del estudio', () => {
  assert.equal(symbolBanner('NVDA', SUMMARY, EVENTS),
    'NVDA: los resultados trimestrales mueven la acción un 5,63 % de media (2 días, sin signo) frente a un 2,23 % de un día normal (×2,5; 27 avisos).');
  const xom = symbolBanner('XOM', SUMMARY, EVENTS);
  assert.match(xom, /×1,0; 27 avisos\)\. Es una excepción: aquí no se distinguen de un día normal\./);
});

test('symbolBanner sin estudio calcula la media de los eventos y sin eventos lo dice', () => {
  assert.match(symbolBanner('NVDA', null, EVENTS), /^NVDA: 2 avisos de resultados; la acción se movió de media un 15,20 % /);
  assert.equal(symbolBanner('PLD', null, []), 'PLD: sin eventos de resultados en los datos.');
});
