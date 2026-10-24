'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { movingAverage } = require('./index');

test('averages consecutive complete windows without changing the input', () => {
  const values = [1, 2, 3, 4, 5];
  assert.deepEqual(movingAverage(values), [2, 3, 4]);
  assert.deepEqual(movingAverage(values, 2), [1.5, 2.5, 3.5, 4.5]);
  assert.deepEqual(values, [1, 2, 3, 4, 5]);
});

test('handles negative numbers, decimals, and boundary window sizes', () => {
  assert.deepEqual(movingAverage([-2, 0, 5], 2), [-1, 2.5]);
  assert.deepEqual(movingAverage([1.5, 2.5], 1), [1.5, 2.5]);
  assert.deepEqual(movingAverage([1, 2, 3], 3), [2]);
});

test('rejects an empty array', () => {
  assert.throws(() => movingAverage([]), {
    name: 'TypeError',
    message: 'values must be a non-empty array',
  });
});

test('rejects non-numeric input', () => {
  for (const values of [null, '123', [1, '2'], [NaN], [Infinity], Array(2)]) {
    assert.throws(() => movingAverage(values, 1), TypeError);
  }
});

test('rejects invalid window sizes', () => {
  for (const size of [0, -1, 1.5, 4, NaN, '2']) {
    assert.throws(() => movingAverage([1, 2, 3], size), RangeError);
  }
});
