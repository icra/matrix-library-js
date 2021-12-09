/*
  Matrix Library in Javascript in a single JS file

  Suppose you have a 3x3 matrix "A":

      | 1 2 1 |
  A = | 3 4 1 |
      | 1 1 1 |

  You can write it as an array of columns, like this:

  let A=[
    [1, 3, 1],
    [2, 4, 1],
    [1, 1, 1]
  ];

  Then you can use the functions of this library to compute several matrix
  operations, for example:

  let A=[ [1, 3, 1], [2, 4, 1], [1, 1, 1] ];
  let d = determinant(A);  //d: -2,
  let T = transposed(A);   //T: [ [ 1, 2, 1 ], [ 3, 4, 1 ], [ 1, 1, 1 ] ],
  let B = multiply(A,A);   //B: [ [ 8, 16, 5 ], [ 11, 23, 7 ], [ 4, 8, 3 ] ],
  let C = escalate(A,5);   //C: [ [ 5, 15, 5 ], [ 10, 20, 5 ], [ 5, 5, 5 ] ],
  let D = sum(A,B);        //D: [ [ 9, 19, 6 ], [ 13, 27, 8 ], [ 5, 9, 4 ] ],
  let E = subtract(A,B);   //E: [ [ -7, -13, -4 ], [ -9, -19, -6 ], [ -3, -7, -2 ] ],
  let F = minor(A,1,2);    //F: [ [ 1, 1 ], [ 2, 1 ] ],
  let G = adjoint(A);      //G: [ [ 3, -1, -2 ], [ -2, 0, 2 ], [ -1, 1, -2 ] ],
  let H = inverse(A);      //H: [ [ -1.5, 1, 0.5 ], [ 0.5, -0, -0.5 ], [ 1, -1, 1 ] ],
  let I = multiply(H,A);   //I: [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ]
  console.log({d,T,B,C,D,E,F,G,H,I});
*/

/*
 Simple assert function
*/
export function assert(expr,message){
  //expr: boolean
  //message: string
  if(!expr) throw(message);
  /*
  assert(1==1,"1 is 1");//passes
  assert(0==1,"0 is not 1");//throws exception
  */
}

/*
  Generate an array of "n" zeros, for example: zeros(3) => [0,0,0]
*/
export function zeros(n){
  let arr=[];
  for(let i=0;i<n;i++) arr.push(0);
  return arr;
  //console.log(zeros(2)); //[0,0]
}

/*
  Generate an array of "n" ones, for example: ones(3) => [1,1,1]
*/
export function ones(n){
  return zeros(n).map(el=>1);
  //console.log(ones(3)); //[1,1,1]
}

/*
  Check that a matrix M is well formed
*/
export function check_matrix(M){
  assert(M.constructor===Array,"Matrix M is not an Array");
  assert(M.length,"Matrix M is empty");
  M.forEach((col,i)=>{
    assert(col.constructor===Array,`Column ${i} is not an Array`);
  });

  //Check that length of columns is the same
  let len = M[0].length;
  assert(len,"Length of the first column is 0");
  M.forEach(col=>{
    assert(col.length==len,"Column length is not the same for all columns");
  });
}

/*
  Check if matrices A and B have the same exact values
*/
export function are_equal(A,B){
  check_matrix(A);
  check_matrix(B);
  assert(A.length==B.length,"A and B have different number of columns");
  assert(A[0].length==B[0].length,"A and B have different number of rows");
  return A.every((col,j)=>col.every((el,i)=>el==B[j][i]));
  /*
  let A=[[1,2],[3,4]];
  let B=[[1,2],[3,4]];
  let C=[[0,0],[0,0]];
  console.log(are_equal(A,B));//true
  console.log(are_equal(A,C));//false
  */
}

/*
  Compute the transposed matrix of M
*/
export function transposed(M){
  check_matrix(M);

  //input matrix "M" is sized [m x n]
  let m = M[0].length; //rows
  let n = M.length;    //cols

  //output matrix "T" will be sized [n x m]
  let T=[];//return value

  for(let i=0;i<m;i++){
    let col=[];
    for(let j=0;j<n;j++){
      col.push(M[j][i]);
    }
    T.push(col);
  }

  return T;
}

/*
  Get the "ith" row; gets the "ith" value from each column
*/
function get_row(M,i){
  assert(i>=0,"Desired row index is negative");
  assert(i<M[0].length,"Desired row index is too large");

  return zeros(M.length).map((el,j)=>M[j][i])
  //console.log(get_row([[1,3],[2,4]],0));
  //console.log(get_row([[1,3],[2,4]],1));
}

/*
  Multiplication of matrices A and B
*/
export function multiply(A,B){
  check_matrix(A);
  check_matrix(B);

  //size of matrix
  let rowsA = A[0].length;
  let colsA = A.length;
  let rowsB = B[0].length;
  let colsB = B.length
  //[rowsA x colsA] x [rowsB x colsB]
  assert(colsA==rowsB,"Number of columns of A and number of rows of B is different");

  let M = []; //return value [m x n]
  let m = rowsA; //rows of M
  let n = colsB; //cols of M
  for(let j=0;j<n;j++){
    let col=[];
    for(let i=0;i<m;i++){
      let val = get_row(A,i).map((el,k)=>el*B[j][k]).reduce((p,c)=>(p+c));
      col.push(val);
    }
    M.push(col);
  }

  return M;
}

/*
  Multiply all elements from a matrix M by an
  escalar (number)
*/
export function escalate(M,escalar){
  check_matrix(M);
  return M.map(col=>col.map(el=>el*escalar));
}

/*
  Sum of matrix A + B
*/
export function sum(A,B){
  check_matrix(A);
  check_matrix(B);
  assert(A.length==B.length,"A and B have different number of columns");
  assert(A[0].length==B[0].length,"A and B have different number of rows");
  return A.map((col,i)=>col.map((el,j)=>el+B[i][j]))
}

/*
  Subtract matrix B from A
*/
export function subtract(A,B){
  return sum(A,escalate(B,-1));
}

/*
  Omit row "i" and column "j" from a matrix M and return resulting minor matrix
*/
export function minor(M,i,j){
  check_matrix(M);
  assert(i>=0 && i<M[0].length,"Wrong number of row");
  assert(j>=0 && j<M.length,"Wrong number of column");

  let N=[];//return value (smaller matrix)

  M.forEach((col,n_col)=>{
    if(n_col==j) return;
    let new_col=[];
    col.forEach((el,n_row)=>{
      if(n_row==i) return;
      new_col.push(el);
    });
    N.push(new_col);
  })

  return N;
}

/*
  Compute determinant of a matrix M
*/
export function determinant(M){
  check_matrix(M);
  assert(M.length==M[0].length,"Matrix M is not square");

  let n = M.length; //number of rows and cols
  if(n==1) return M[0][0];
  if(n==2) return M[0][0]*M[1][1] - M[0][1]*M[1][0];

  let det = 0;//initialize determinant at 0

  //do recursive calls using minors
  for(let i=0;i<n;i++){
    det += M[0][i]*Math.pow(-1,i+2)*determinant(minor(M,i,0));
  }

  return det;
  //determinant([[1,3],[2,4]]); //-2
}

/*
  Compute the adjoint of a matrix M
*/
export function adjoint(M){
  check_matrix(M);

  let A=[];//return value
  M.forEach((col,j)=>{
    let new_col=col.map((el,i)=>{
      return Math.pow(-1,i+j+2)*determinant(minor(M,i,j));
    });
    A.push(new_col);
  });
  return A;
  //let A=[[1,0,4],[0,3,0],[2,0,5]];
  //console.log(adjoint(A)); //[[15,0,-6],[0,-3,0],[-12,0,3]]
}

/*
  Compute the inverse of a matrix M
*/
export function inverse(M){
  check_matrix(M);
  let d = determinant(M);
  assert(d,"Determinant from matrix M is zero");
  //adjoint de la transposed
  let Mta = adjoint(transposed(M));
  return escalate(Mta,1/d);
  //let A=[[4,0,1],[0,0,-2],[0,-2,8]];
  //console.log(multiply(A,inverse(A)));//identity
}

