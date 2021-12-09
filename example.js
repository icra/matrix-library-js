import * as M from './module.js';

let A  = [[1,3],[2,4]];
let Ai = M.inverse(A);
let I  = M.multiply(A,Ai);

console.log({A,Ai,I});
