status: usable but not everything documented [work in progress as 2022-11-25]

Matrix Library in Javascript in a single JS file

Suppose you have a 3x3 matrix called "A":

```
    | 1 2 1 |
A = | 3 4 1 |
    | 1 1 1 |
```

You can write it as an array of columns, like this:

```javascript
let A=[
  [1, 3, 1],
  [2, 4, 1],
  [1, 1, 1]
];
```

Then you can use the functions of this library to compute several matrix
operations, for example:

```javascript
import * as M from './module.js';

let A=[ [1, 3, 1], [2, 4, 1], [1, 1, 1] ];
let d = M.determinant(A);  //d: -2,
let T = M.transposed(A);   //T: [ [ 1, 2, 1 ], [ 3, 4, 1 ], [ 1, 1, 1 ] ],
let B = M.multiply(A,A);   //B: [ [ 8, 16, 5 ], [ 11, 23, 7 ], [ 4, 8, 3 ] ],
let C = M.escalate(A,5);   //C: [ [ 5, 15, 5 ], [ 10, 20, 5 ], [ 5, 5, 5 ] ],
let D = M.sum(A,B);        //D: [ [ 9, 19, 6 ], [ 13, 27, 8 ], [ 5, 9, 4 ] ],
let E = M.subtract(A,B);   //E: [ [ -7, -13, -4 ], [ -9, -19, -6 ], [ -3, -7, -2 ] ],
let F = M.minor(A,1,2);    //F: [ [ 1, 1 ], [ 2, 1 ] ],
let G = M.adjoint(A);      //G: [ [ 3, -1, -2 ], [ -2, 0, 2 ], [ -1, 1, -2 ] ],
let H = M.inverse(A);      //H: [ [ -1.5, 1, 0.5 ], [ 0.5, -0, -0.5 ], [ 1, -1, 1 ] ],
let I = M.multiply(H,A);   //I: [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ]
console.log({d,T,B,C,D,E,F,G,H,I});
```
