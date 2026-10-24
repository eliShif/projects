# Moving average

`movingAverage(values, windowSize = 3)` returns the arithmetic mean of each
complete consecutive window, leaving the input unchanged.

```js
const { movingAverage } = require('./index');
movingAverage([1, 2, 3, 4], 2); // [1.5, 2.5, 3.5]
```

Input must be a non-empty array of finite numbers. The window size must be an
integer between 1 and the array length. Empty arrays and invalid numeric input
throw `TypeError`; invalid window sizes throw `RangeError`.

Run the tests using Node.js with built-in test runner support:

```sh
node --test index.test.js
```
