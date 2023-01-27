/*
  Matrix Library in Javascript in a single JS file

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
  Acumulative inverse distribution F (equivalent to Matlab's "finv", or R's "qf")
*/
import {finv} from "./finv/finv.js"

/*
  TOL: aproximation to zero
*/
const TOL = 1e-40;

/*
  Array utils
*/
Array.prototype.sum=function(){return this.reduce((p,c)=>(p+c),0)};//number
Array.prototype.mean=function(){return this.length?this.sum()/this.length:0};//number
Array.prototype.stddev=function(){
  let n = this.length;//number
  if(n<2) return 0;//number
  let m = this.mean();//number
  let square_diffs = this.map(x=>Math.pow(x-m,2));//array
  let ss = square_diffs.sum();//number
  return Math.sqrt(ss/(n-1));//number
};
Array.prototype.mean_center=function(){
  let m = this.mean();//number
  return this.map(x=>(x-m));//array
}
Array.prototype.normalize=function(){
  let m = this.mean();//number
  let s = this.stddev();//number
  return this.map(x=>(x-m)/s);//array
}

/*
  Assert function
*/
export function assert(expr,message){
  //expr: boolean
  //message: string
  /*
    examples:
    assert(1==1,"1 is 1");//passes
    assert(0==1,"0 is not 1");//throws exception
  */
  if(!expr) throw(message);
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

  //check that length of each column is the same
  let len = M[0].length;
  assert(len,"Length of the first column is 0");
  M.forEach(col=>{
    assert(col.length==len,"Column length is not the same for all columns");
  });
}

/*
  Calculate size of matrix M (rows, columns)
*/
export function size_of_matrix(M){
  check_matrix(M);
  let rows = M[0].length;
  let cols = M.length;
  return {rows,cols};
}

/*
  Create an array of "n" zeros.
    - example: zeros(3) => [0,0,0]
*/
export function zeros(n){
  let arr=new Array(n);
  for(let i=0;i<n;i++){
    arr[i]=0;
  }
  return arr;
}

/*
  Create an array of "n" ones.
    - example: ones(3) => [1,1,1]
*/
export function ones(n){
  return zeros(n).map(x=>1);
}

/*
  Check if matrix A and matrix B have the same exact values
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
  Multiplication of matrix A and matrix B
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
  Multiply all matrix M's elements by a scalar (number)
*/
export function escalate(M,escalar){
  check_matrix(M);
  return M.map(col=>col.map(n=>n*escalar));
}

/*
  Sum of matrix A + matrix B
*/
export function sum(A,B){
  check_matrix(A);
  check_matrix(B);
  assert(A.length==B.length,"A and B have different number of columns");
  assert(A[0].length==B[0].length,"A and B have different number of rows");
  return A.map((col,i)=>col.map((el,j)=>el+B[i][j]))
}

/*
  Subtraction of matrix A - matrix B
*/
export function subtract(A,B){
  return sum(A,escalate(B,-1));
}

/*
  Create a new "minor" matrix from matrix M:
  omitting row "i" and column "j"
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
  Determinant of matrix M
*/
export function determinant(M){
  check_matrix(M);

  //mandatory: matrix has to be square
  assert(M.length==M[0].length,"Matrix M is not square");

  //number of rows and cols
  let n = M.length;

  if(n==1) return M[0][0];
  if(n==2) return M[0][0]*M[1][1] - M[0][1]*M[1][0];

  //transform M to triangular matrix T
  let T=gaussian_elimination(M);

  //initialize determinant at 1
  let det=1;

  //multiply all main diagonal numbers
  for(let i=0;i<n;i++){
    det *= T[i][i];
  }

  return det;
}

/*
  Adjoint of matrix M
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
  Inverse of matrix M
*/
export function inverse(M){
  check_matrix(M);
  let d=determinant(M);
  assert(d,"Matrix cannot be inverted, determinant is zero");

  //adjoint of transposed of M
  let Mta = adjoint(transposed(M));
  return escalate(Mta,1/d);

  //test
  //let A=[[4,0,1],[0,0,-2],[0,-2,8]];
  //console.log(multiply(A,inverse(A)));//identity
}

/*
  Create an identity matrix of size n
*/
export function identity(n){
  let I=new Array(n);
  for(let i=0;i<n;i++){
    I[i]=zeros(n);
  }

  for(let i=0;i<n;i++){
    I[i][i]=1;
  };

  return I;
  //test
  //console.log(identity(0));
  //console.log(identity(1));
  //console.log(identity(2));
  //console.log(identity(3));
  //console.log(identity(4));
}

/*
  Gaussian Elimination
  Transform matrix M to triangular matrix
*/
export function gaussian_elimination(M){
  check_matrix(M);

  let n = M.length; //size

  //create a copy of M
  let T = M.map(col=>col.map(x=>x));

  if(n<2) return T;

  //iterate rows (start on 2nd row)
  for(let i=1;i<n;i++){
    if(T[i][0]==0) continue;

    //the first element of the diagonal cannot be 0
    //try to make a combination with another row
    if(T[0][0]==0){
      //next row
      let k=1;
      while(true){
        if(T[k][0]){
          //modify the entire row
          for(let j=0;j<n;j++){
            T[0][j] = T[0][j] + T[k][j];
          }
          break;
        }
        k++;
      }
    }

    //now T[0][0] is guaranteed not being zero
    let ratio = -T[i][0]/T[0][0];

    //modify the entire row, so that T[i][0] is 0
    T[i][0]=0;
    for(let j=1;j<n;j++){
      T[i][j] = T[i][j] + ratio*T[0][j];
    }
  }

  //now all column is zero except first element
  //we can recursive call using the minor
  let t = gaussian_elimination(minor(T,0,0));
  for(let i=0;i<n-1;i++){
    for(let j=0;j<n-1;j++){
      T[i+1][j+1]=t[i][j];
    }
  }

  return T;
}

/*
  Mean center and scale with unit variance each column of matrix M
*/
export function normalize_matrix(X){
  return X.map(col=>col.normalize());
}

/*
  Compute covariance matrix
*/
export function covariance_matrix(X){
  check_matrix(X);

  //number of observations
  let n = X[0].length;

  //mean center every column
  let M = X.map(col=>col.mean_center());

  //covariance matrix
  let S = escalate( multiply(transposed(M),M), 1/(n-1));
  return S;
}

/*
  Check if "eig" is an eigenvalue of matrix A
*/
export function check_eigenvalue(A,eig){
  //A: matrix
  //eig: number
  let n = A.length;
  let diag_eig = escalate(identity(n),eig); //diagonal matrix with eig
  let det = determinant(subtract(A,diag_eig)); //det(A-λ·I)
  return Math.abs(det)<TOL;
}

/** SVD procedure as explained in "Singular Value Decomposition and Least Squares Solutions. By G.H. Golub et al."
  *
  * This procedure computes the singular values and complete orthogonal decomposition of a real rectangular matrix A:
  *    A = U * diag(q) * V(t), U(t) * U = V(t) * V = I
  * where the arrays a, u, v, q represent A, U, V, q respectively. The actual parameters corresponding to a, u, v may
  * all be identical unless withu = withv = {true}. In this case, the actual parameters corresponding to u and v must
  * differ. m >= n is assumed (with m = a.length and n = a[0].length)
  *
  *  @param a {Array} Represents the matrix A to be decomposed
  *  @param [withu] {bool} {true} if U is desired {false} otherwise
  *  @param [withv] {bool} {true} if U is desired {false} otherwise
  *  @param [eps] {Number} A constant used in the test for convergence; should not be smaller than the machine precision
  *  @param [tol] {Number} A machine dependent constant which should be set equal to B/eps0 where B is the smallest
  *    positive number representable in the computer
  *
  *  @returns {Object} An object containing:
  *    q: A vector holding the singular values of A; they are non-negative but not necessarily ordered in
  *      decreasing sequence
  *    u: Represents the matrix U with orthonormalized columns (if withu is {true} otherwise u is used as
  *      a working storage)
  *    v: Represents the orthogonal matrix V (if withv is {true}, otherwise v is not used)
  *
*/
export function SVD(a, withu, withv, eps, tol){
  // Define default parameters
  withu = withu !== undefined ? withu : true
  withv = withv !== undefined ? withv : true
  eps = eps || Math.pow(2, -52)
  tol = 1e-64 / eps

  // throw error if a is not defined
  if(!a) {
    throw new TypeError('Matrix a is not defined')
  }

  // Householder's reduction to bidiagonal form
  const n = a[0].length
  const m = a.length
  if (m < n) {
    throw new TypeError('Invalid matrix: m < n')
  }
  let i, j, k, l, l1, c, f, g, h, s, x, y, z
  g = 0
  x = 0
  const e = []
  const u = []
  const v = []
  const mOrN = (withu === 'f') ? m : n

  // Initialize u
  for (i = 0; i < m; i++) {
    u[i] = new Array(mOrN).fill(0)
  }
  // Initialize v
  for (i = 0; i < n; i++) {
    v[i] = new Array(n).fill(0)
  }

  // Initialize q
  const q = new Array(n).fill(0)
  // Copy array a in u
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      u[i][j] = a[i][j]
    }
  }
  for (i = 0; i < n; i++) {
    e[i] = g
    s = 0
    l = i + 1
    for (j = i; j < m; j++) {
      s += Math.pow(u[j][i], 2)
    }
    if (s < tol) {
      g = 0
    } else {
      f = u[i][i]
      g = f < 0 ? Math.sqrt(s) : -Math.sqrt(s)
      h = f * g - s
      u[i][i] = f - g
      for (j = l; j < n; j++) {
        s = 0
        for (k = i; k < m; k++) {
          s += u[k][i] * u[k][j]
        }
        f = s / h
        for (k = i; k < m; k++) {
          u[k][j] = u[k][j] + f * u[k][i]
        }
      }
    }
    q[i] = g
    s = 0
    for (j = l; j < n; j++) {
      s += Math.pow(u[i][j], 2)
    }
    if (s < tol) {
      g = 0
    } else {
      f = u[i][i + 1]
      g = f < 0 ? Math.sqrt(s) : -Math.sqrt(s)
      h = f * g - s
      u[i][i + 1] = f - g
      for (j = l; j < n; j++) {
        e[j] = u[i][j] / h
      }
      for (j = l; j < m; j++) {
        s = 0
        for (k = l; k < n; k++) {
          s += u[j][k] * u[i][k]
        }
        for (k = l; k < n; k++) {
          u[j][k] = u[j][k] + s * e[k]
        }
      }
    }
    y = Math.abs(q[i]) + Math.abs(e[i])
    if (y > x) {
      x = y
    }
  }
  // Accumulation of right-hand transformations
  if (withv) {
    for (i = n - 1; i >= 0; i--) {
      if (g !== 0) {
        h = u[i][i + 1] * g
        for (j = l; j < n; j++) {
          v[j][i] = u[i][j] / h
        }
        for (j = l; j < n; j++) {
          s = 0
          for (k = l; k < n; k++) {
            s += u[i][k] * v[k][j]
          }
          for (k = l; k < n; k++) {
            v[k][j] = v[k][j] + s * v[k][i]
          }
        }
      }
      for (j = l; j < n; j++) {
        v[i][j] = 0
        v[j][i] = 0
      }
      v[i][i] = 1
      g = e[i]
      l = i
    }
  }
  // Accumulation of left-hand transformations
  if (withu) {
    if (withu === 'f') {
      for (i = n; i < m; i++) {
        for (j = n; j < m; j++) {
          u[i][j] = 0
        }
        u[i][i] = 1
      }
    }
    for (i = n - 1; i >= 0; i--) {
      l = i + 1
      g = q[i]
      for (j = l; j < mOrN; j++) {
        u[i][j] = 0
      }
      if (g !== 0) {
        h = u[i][i] * g
        for (j = l; j < mOrN; j++) {
          s = 0
          for (k = l; k < m; k++) {
            s += u[k][i] * u[k][j]
          }
          f = s / h
          for (k = i; k < m; k++) {
            u[k][j] = u[k][j] + f * u[k][i]
          }
        }
        for (j = i; j < m; j++) {
          u[j][i] = u[j][i] / g
        }
      } else {
        for (j = i; j < m; j++) {
          u[j][i] = 0
        }
      }
      u[i][i] = u[i][i] + 1
    }
  }
  // Diagonalization of the bidiagonal form
  eps = eps * x
  let testConvergence
  for (k = n - 1; k >= 0; k--) {
    for (let iteration = 0; iteration < 50; iteration++) {
      // test-f-splitting
      testConvergence = false
      for (l = k; l >= 0; l--) {
        if (Math.abs(e[l]) <= eps) {
          testConvergence = true
          break
        }
        if (Math.abs(q[l - 1]) <= eps) {
          break
        }
      }
      if (!testConvergence) { // cancellation of e[l] if l>0
        c = 0
        s = 1
        l1 = l - 1
        for (i = l; i < k + 1; i++) {
          f = s * e[i]
          e[i] = c * e[i]
          if (Math.abs(f) <= eps) {
            break // goto test-f-convergence
          }
          g = q[i]
          q[i] = Math.sqrt(f * f + g * g)
          h = q[i]
          c = g / h
          s = -f / h
          if (withu) {
            for (j = 0; j < m; j++) {
              y = u[j][l1]
              z = u[j][i]
              u[j][l1] = y * c + (z * s)
              u[j][i] = -y * s + (z * c)
            }
          }
        }
      }
      // test f convergence
      z = q[k]
      if (l === k) { // convergence
        if (z < 0) {
          // q[k] is made non-negative
          q[k] = -z
          if (withv) {
            for (j = 0; j < n; j++) {
              v[j][k] = -v[j][k]
            }
          }
        }
        break // break out of iteration loop and move on to next k value
      }
      // Shift from bottom 2x2 minor
      x = q[l]
      y = q[k - 1]
      g = e[k - 1]
      h = e[k]
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y)
      g = Math.sqrt(f * f + 1)
      f = ((x - z) * (x + z) + h * (y / (f < 0 ? (f - g) : (f + g)) - h)) / x
      // Next QR transformation
      c = 1
      s = 1
      for (i = l + 1; i < k + 1; i++) {
        g = e[i]
        y = q[i]
        h = s * g
        g = c * g
        z = Math.sqrt(f * f + h * h)
        e[i - 1] = z
        c = f / z
        s = h / z
        f = x * c + g * s
        g = -x * s + g * c
        h = y * s
        y = y * c
        if (withv) {
          for (j = 0; j < n; j++) {
            x = v[j][i - 1]
            z = v[j][i]
            v[j][i - 1] = x * c + z * s
            v[j][i] = -x * s + z * c
          }
        }
        z = Math.sqrt(f * f + h * h)
        q[i - 1] = z
        c = f / z
        s = h / z
        f = c * g + s * y
        x = -s * g + c * y
        if (withu) {
          for (j = 0; j < m; j++) {
            y = u[j][i - 1]
            z = u[j][i]
            u[j][i - 1] = y * c + z * s
            u[j][i] = -y * s + z * c
          }
        }
      }
      e[l] = 0
      e[k] = f
      q[k] = x
    }
  }
  // Number below eps should be zero
  for (i = 0; i < n; i++) {
    if (q[i] < eps) q[i] = 0
  }
  return { u, q, v };
}

/*
  Create a square diagonal matrix from an array
*/
export function diag(arr){
  let n = arr.length;
  let M = new Array(n)
  for(let i=0;i<n;i++) M[i]    = zeros(n);
  for(let i=0;i<n;i++) M[i][i] = arr[i];
  return M;
}

/*
  Compute PCA of matrix M
  - Also calculate Hotelling T2 and Q faults and contributions of each variable
  - Thresholds for Q and T2 are computed using 95% confidence
*/
export function PCA(M, a, normalize, change_sign_of_loadings){
  check_matrix(M);

  //parameters
  a=a||false; //nº of PCs to keep (or automatic)
  normalize=normalize??true; //true if undefined
  change_sign_of_loadings=change_sign_of_loadings??false; //false if undefined

  let m = M.length;    //nº of variables
  let n = M[0].length; //nº of observations

  let X; //new matrix X with centered data
  if(normalize){
    //mean center AND scale (variance=1)
    X = M.map(col=>col.normalize());
  }else{
    //mean center only
    X = M.map(col=>col.mean_center());
  }

  //compute S:
  //  - S is the correlation matrix (if X is normalized)
  //  - S is the covariance matrix (if X is mean centered only)
  let S = escalate(multiply(transposed(X),X),1/(n-1));

  //SVD and eigenvalues of S
  let svd                  = SVD(S); //object {q,u,v}
  let eigenvalues_unsorted = svd.q.map(n=>n); //array length m
  let loadings_unsorted    = transposed(svd.u); //matrix size mxm

  //sort eigenvalues and loadings
  let eigenvalues_sorted = [];
  let loadings_sorted    = [];
  for(let i=0; i<eigenvalues_unsorted.length; i++){
    let max   = Math.max.apply(null, eigenvalues_unsorted);
    let index = eigenvalues_unsorted.indexOf(max);
    eigenvalues_sorted.push(eigenvalues_unsorted[index]);
    let pc = loadings_unsorted[index];
    loadings_sorted.push(pc);
    eigenvalues_unsorted[index]=-Infinity;
  }

  let sum_eigs             = eigenvalues_sorted.sum();
  let variance             = eigenvalues_sorted.map((e,i)=>100*e/sum_eigs);    //percentages
  let accumulated_variance = variance.map((e,i)=>variance.slice(0,i+1).sum()); //percentages (accumulated)

  //we keep "a" loadings: number of principal components
  let info=[];
  if(a===false){ //calculate a
    for(let i=0;i<accumulated_variance.length;i++){
      if(accumulated_variance[i]>99){
        a=i+1; //number of PCs to keep
        break;
      }
    }
    info.push(`Number of PCs kept: ${a} <-- determined automatically`);
  }else{
    info.push(`Number of PCs kept: ${a} <-- passed as parameter`);
  }
  info.push(`Kept ${a} of ${m} PCs: explaining ${(accumulated_variance[a-1]).toFixed(2)}% of variance`);
  //console.log(info);

  //eigenvalues not considered: keep them for later analysis (Q test)
  let eigenvalues_not_considered = eigenvalues_sorted.slice(a);
  //console.log({eigenvalues_not_considered});

  //overwrite loadings and eigenvalues
  let loadings    = loadings_sorted.slice(0,a); //only the first "a" columns
  let eigenvalues = eigenvalues_sorted.slice(0,a);

  //multiply loadings by -1, if selected
  if(change_sign_of_loadings){
    loadings = escalate(loadings,-1);
  }

  //compute scores: observations in the new space
  let scores = multiply(X,loadings);

  //compute residual matrix
  let residuals = subtract(X, multiply(scores,transposed(loadings)));
  //-----------------------------------------------------------------

  //Q (also known as "SPE") contribution analysis
  let Q_threshold_95 = (function(){
    let sqrt   = Math.sqrt; //function
    let theta1 = eigenvalues_not_considered.sum(); //number
    let theta2 = eigenvalues_not_considered.map(e=>e**2).sum(); //number
    let theta3 = eigenvalues_not_considered.map(e=>e**3).sum(); //number
    if(theta1==0) return Infinity;

    //llindar per Q amb una significança del 5%
    let ca = 1.644854; //from qnorm(0.95,0,1) in R; //number
    let h0 = 1-(2*theta1*theta3)/(3*theta2**2);
    return theta1*(ca*h0*sqrt(2*theta2)/theta1 + 1 + theta2*h0*(h0-1)/(theta1**2))**(1/h0); //number
  })(); //number
  let rrT = multiply(residuals,transposed(residuals)); //matrix nxn
  let Q   = rrT.map((col,i)=>col[i]); //array (diagonal of rrT)
  let RESS = Q.sum(); //residual sum of squares

  //observations with faults for Q
  let faults_for_Q=[];
  Q.forEach((n,i)=>{
    let alarm = n >= Q_threshold_95;
    let Q_residual = n;
    let contributions = get_row(residuals,i);
    faults_for_Q.push({
      observation:i,
      Q_residual,
      contributions,
      alarm,
    });
  });

  //Hotelling T^2 contribution analysis
  let T2_threshold_95 = a*(n+1)*(n-1)/(n*(n-a))*finv(0.95,a,n-a);
  let T2_by_observation=[];
  for(let i=0;i<n;i++){
    //observation i reprojected into principal components
    let t = transposed([get_row(scores,i)]);
    //T2 for this observation
    let T2 = multiply(multiply(t,diag(eigenvalues.map(e=>1/e))),transposed(t))[0][0];
    T2_by_observation.push(T2);
  }

  //observations with faults for T2
  let faults_for_T2=[];
  T2_by_observation.forEach((n,i)=>{
    let T2_residual = n; //number
    let alarm       = n >= T2_threshold_95; //bool

    //get observation i (faulty)
    let xi = get_row(X,i); //array of size m

    //scores of observation i (faulty)
    let ti = get_row(scores,i); //array of size a

    //contribution from each "original" variable
    let contributions=new Array(m); //array of size m

    for(let j=0;j<m;j++){//forEach variable
      let cont = 0;
      for(let k=0;k<a;k++){//forEach PC
        cont += ti[k]/eigenvalues[k]*loadings[k][j]*xi[j];
      }
      contributions[j] = cont;
    }

    //console.log({contributions});
    faults_for_T2.push({
      observation:i,
      T2_residual,
      contributions,
      alarm,
    });
  });

  return{
    observations:n,
    variables:m,
    info,

    input_matrix:X,

    eigenvalues,
    eigenvalues_sorted,
    variance,
    accumulated_variance,

    loadings,
    scores,
    residuals,
    RESS,

    //SPC data (control charts)
    Q_threshold_95,
    T2_threshold_95,
    Q_by_observation:Q,
    T2_by_observation,
    faults_for_Q,
    faults_for_T2,
  };
}

/*create PCA report (CSV)*/
/* idea: node file.js > result.csv */
export function create_PCA_report(X,a,names_columns){
  //X: numeric matrix
  //a: number of PCs to keep (optional)
  //names_columns: array of strings
  let csv_report="";//string

  //object with the results of the PCA() call
  let res = PCA(X,a);

  //print EIGENVALUES FOR SCREE PLOT (determine visually the number of PCs to keep)
  csv_report+="\n,"                              +res.eigenvalues_sorted.map((e,i)=>`PC${i+1}`).join(',')+'\n';
  csv_report+="eigenvalues (most important PCs),"+res.eigenvalues.join(',')+'\n';
  csv_report+="eigenvalues (all PCs),"           +res.eigenvalues_sorted.join(',')+'\n';
  csv_report+="variance (all PCs) (%),"          +res.variance.join(',')+'\n';
  csv_report+="acc. variance (all PCs) (%),"     +res.accumulated_variance.join(',')+'\n';

  //print LOADINGS
  csv_report+="\nLOADINGS\n";
  res.loadings.forEach((row,i)=>{
    if(i==0) csv_report+='PC,'+names_columns.join(',')+'\n';
    csv_report+=`PC${i+1},`+row.join(',')+'\n';
  });

  //print SCORES (PCs) next to ORIGINAL DATA
  csv_report+="\nSCORES and ORIGINAL DATA\n";
  let scores        = transposed(res.scores);
  let original_data = transposed(X);
  for(let i=0; i<res.observations; i++){
    if(i==0) csv_report+='observation,'+scores[i].map((n,i)=>`PC${i+1}`).join(',')+','+names_columns.join(',')+'\n';
    csv_report+=String(i+1)+','+scores[i].join(',')+','+original_data[i].join(',')+'\n';
  };

  /*FAULT DETECTION section*/
  //table Q vs T2
  csv_report+="\nobservation,Q,Q_threshold_95,T2,T2_threshold_95\n";
  for(let i=0;i<res.observations;i++){
    csv_report+=`${i+1},${res.Q_by_observation[i]},${res.Q_threshold_95},${res.T2_by_observation[i]},${res.T2_threshold_95}\n`;
  }

  //print contributions to faults by variable
  csv_report+="\nFAULTS DETECTED contribution by variable\n";
  csv_report+=`Q_threshold_95, ${res.Q_threshold_95}\n`;
  csv_report+=`T2_threshold_95, ${res.T2_threshold_95}\n`;

  res.faults_for_Q.forEach((f,i)=>{
    if(i==0) csv_report+='\nQ_residual,observation,'+names_columns.join(',')+'\n';
    csv_report+=`${f.Q_residual},${f.observation+1},`+f.contributions.join(',')+'\n';
  });

  res.faults_for_T2.forEach((f,i)=>{
    if(i==0) csv_report+='\nT2_residual,observation,'+names_columns.join(',')+'\n';
    csv_report+=`${f.T2_residual},${f.observation+1},`+f.contributions.join(',')+'\n';
  });

  console.log(csv_report);
  return csv_report;
}
