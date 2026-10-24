'use strict';

/** Return numeric means for complete consecutive windows without mutating values. */
function movingAverage(values, windowSize = 3) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new TypeError('values must be a non-empty array');
  }
  for (const value of values) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new TypeError('values must contain only finite numbers');
    }
  }
  if (!Number.isInteger(windowSize) || windowSize < 1 || windowSize > values.length) {
    throw new RangeError('windowSize must be an integer from 1 to values.length');
  }

  const averages = [];
  for (let start = 0; start <= values.length - windowSize; start++) {
    let average = 0;
    for (let offset = 0; offset < windowSize; offset++) {
      average += values[start + offset] / windowSize;
    }
    averages.push(average);
  }
  return averages;
}

module.exports = { movingAverage };
