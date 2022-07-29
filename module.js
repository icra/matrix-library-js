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

//aproximation to zero
const TOL=1e-40;

/* Numeric array utils */
Array.prototype.sum=function(){return this.reduce((p,c)=>(p+c),0)} //number
Array.prototype.mean=function(){return this.length?this.sum()/this.length:0}; //number
Array.prototype.stddev=function(){
  let n = this.length; //number
  if(n<2) return 0; //number
  let m = this.mean(); //number
  let square_diffs = this.map(x=>Math.pow(x-m,2)); //array
  let ss = square_diffs.sum(); //number
  return Math.sqrt(ss/(n-1)); //number
};
Array.prototype.mean_center=function(){
  let m = this.mean(); //number
  return this.map(x=>(x-m)); //array
}
Array.prototype.normalize=function(){
  let m = this.mean(); //number
  let s = this.stddev(); //number
  return this.map(x=>(x-m)/s); //array
}

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

export function size_of_matrix(M){
  check_matrix(M);
  let rows = M[0].length;
  let cols = M.length;
  return {rows,cols};
}

/*
  Generate an array of "n" zeros, for example: zeros(3) => [0,0,0]
*/
export function zeros(n){
  let arr=new Array(n);
  for(let i=0;i<n;i++){
    arr[i]=0;
  }
  return arr;
  //console.log(zeros(2))
}

/*
  Generate an array of "n" ones, for example: ones(3) => [1,1,1]
*/
export function ones(n){
  return zeros(n).map(el=>1);
  //console.log(ones(3))
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

  //transform matrix to triangular
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
  assert(d,"Matrix M cannot be inverted, determinant is zero");

  //adjoint de la transposed
  let Mta = adjoint(transposed(M));
  return escalate(Mta,1/d);
  //let A=[[4,0,1],[0,0,-2],[0,-2,8]];
  //console.log(multiply(A,inverse(A)));//identity
}

/*
  generate an identity matrix of size nxn
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
  //console.log(identity(0));
  //console.log(identity(1));
  //console.log(identity(2));
  //console.log(identity(3));
  //console.log(identity(4));
}

/*
  Gaussian Elimination
  Transform matrix M to triangle matrix T)
*/
export function gaussian_elimination(M){
  check_matrix(M);

  let n = M.length; //size

  //copy M to T (new matrix)
  let T = M.map(col=>col.map(el=>el));

  if(n==1) return T;

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

    //now T[0][0] is not zero
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

//mean center and scaled with unit variance
export function normalize_matrix(X){
  return X.map(col=>col.normalize());
}

//compute covariance matrix
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

//create a square diagonal matrix from an array
export function diag(arr){
  let n = arr.length;
  let M = new Array(n)
  for(let i=0;i<n;i++) M[i]    = zeros(n);
  for(let i=0;i<n;i++) M[i][i] = arr[i];
  return M;
}

export function PCA(M){
  let m = M.length;    //nº of variables
  let n = M[0].length; //nº of observations
  console.log("===PCA===")
  console.log({observations:n, variables:m});

  //mean center and scale with unit variance
  let X = M.map(col=>col.normalize());

  //covariance matrix
  let S = escalate(multiply(transposed(X),X),1/(n-1));

  //SVD and eigenvalues
  let svd                  = SVD(S);
  let eigenvalues_unsorted = svd.q; //array length m
  let loadings_unsorted    = transposed(svd.u); //matrix size mxm

  //sort eigenvalues and loadings
  let eigenvalues = [];
  let loadings    = [];
  for(let i=0; i<eigenvalues_unsorted.length; i++){
    let max   = Math.max.apply(null, eigenvalues_unsorted);
    let index = eigenvalues_unsorted.indexOf(max);
    eigenvalues.push(eigenvalues_unsorted[index]);
    let pc = loadings_unsorted[index];
    loadings.push(pc);
    eigenvalues_unsorted[index]=-Infinity;
  }

  let sum_eigs             = eigenvalues.sum();
  let accumulated_variance = eigenvalues.map((e,i)=>eigenvalues.slice(0,i+1).sum()/sum_eigs);
  console.log({eigenvalues,accumulated_variance});

  //we keep "a" loadings: number of principal components
  let a = m;
  for(let i=0;i<accumulated_variance.length;i++){
    if(accumulated_variance[i]>0.99){
      a = i+1;
      break;
    }
  }
  console.log(`[info] Keeping ${a} of ${m} PCs explaining ${(100*accumulated_variance[a-1]).toFixed(2)}% of variance`);

  //eigenvalues not considered for later analysis (Q test)
  let eigenvalues_not_considered = eigenvalues.slice(a);
  //console.log({eigenvalues_not_considered});

  //overwrite loadings and eigenvalues
  loadings    = loadings.slice(0,a); //només "a" columnes
  eigenvalues = eigenvalues.slice(0,a);

  loadings.forEach((pc,i)=>{
    console.log(`PC${i+1}, ${pc.map(n=>n.toFixed(2)).join(', ')}`);
  });

  //compute scores: observations in the new space
  let scores = multiply(X,loadings);

  //compute residual matrix
  let residuals = subtract(X, multiply(scores,transposed(loadings)));
  //-----------------------------------------------------------------

  //SPE Q contribution analysis
  let Q_threshold_95 = (function(){

    let sqrt      = Math.sqrt; //function
    let theta1    = eigenvalues_not_considered.sum(); //number
    let theta2    = eigenvalues_not_considered.map(e=>e**2).sum(); //number
    let theta3    = eigenvalues_not_considered.map(e=>e**3).sum(); //number

    if(theta1==0) return Infinity;

    //llindar per Q amb una significança del 5%
    let ca        = 1.644854; //from qnorm(0.95,0,1) in R; //number
    let h0        = 1-(2*theta1*theta3)/(3*theta2**2);
    return theta1*(ca*h0*sqrt(2*theta2)/theta1 + 1 + theta2*h0*(h0-1)/(theta1**2))**(1/h0); //number
  })(); //number
  let rrT = multiply(residuals,transposed(residuals)); //matrix nxn
  let Q   = rrT.map((col,i)=>col[i]); //array (diagonal of rrT)

  //observations with faults for Q
  let faults_for_Q=[];
  Q.forEach((n,i)=>{
    if(n<Q_threshold_95) return;
    let Q_residual = n;
    let contributions = get_row(residuals,i);
    faults_for_Q.push({observation:i, Q_residual, contributions});
  });
  console.log({Q_threshold_95,faults_for_Q});
  //-----------------------------------------------------------------

  //Hotelling T^2 contribution analysis
  let T2_threshold_95 = a*(n-1)/(n-a)*finv(0.95,a,n-a);
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
    if(n<T2_threshold_95) return;
    let T2_residual = n;

    //observation i reprojected into principal components
    let t = get_row(scores,i); //array of size a

    //contribution from each pc
    let contributions=[];
    for(let j=0;j<a;j++){
      let pij   = loadings[j][i]; //number
      let ti    = t[j]; //number
      let eig_i = eigenvalues[j]; //number
      let xj    = get_row(X,i); //array
      let cont  = xj.map(x=> ti/eig_i*pij*x); //array
      contributions.push(cont);
    }

    //transpose to be able to sum each variable
    contributions = transposed(contributions).map(col=>col.sum());
    console.log({contributions});
    faults_for_T2.push({observation:i, T2_residual, contributions});
  });

  //find observations with fauls for Q
  console.log({T2_threshold_95,faults_for_T2});

  //print table T2 vs Q
  (function(){
    return;
    console.log("T2, Q, (first row are thresholds)");
    console.log(`${T2_threshold_95}, ${Q_threshold_95}`);
    for(let i=0;i<n;i++){
      console.log(`${T2_by_observation[i]}, ${Q[i]}`);
    }
  })();
}

//test
const M=[
  [22, 10, 2, 3, 7],
  [14, 7, 10, 0, 8],
  [-1, 13, -1, -11, 3],
  [-3, -2, 13, -2, 4],
  [9, 8, 1, -2, 4],
  [9, 1, -7, 5, -1],
  [2, -6, 6, 5, 1],
  [4, 5, 0, -2, 2]
];
//PCA(M);

const M2=[
  [10, 121.9   , 120.9   , 124.1   , 122.2   , 119.8   , 125.745 , 123.43  , 123.883 , 125.353 , 126.0153 , 125.3861 , 124.4702 , 121.3318 , 123.4016 , 121.654] ,
  [10, 91.137  , 94.685  , 90.322  , 90.025  , 86.933  , 89.343  , 89.125  , 94.878  , 89.303  , 93.7609  , 92.4001  , 90.0061  , 87.8128  , 88.6698  , 85.324]  ,
  [10, 108.096 , 111.838 , 107.391 , 106.646 , 103.966 , 105.897 , 107.203 , 107.497 , 107.976  , 109.373 , 111.612  , 109.7599 , 105.9164 , 110.0603 , 110.419] ,
  [10, 126.2   , 124.3   , 126.9   , 125.9   , 123.8   , 129.197 , 129.044 , 126.674 , 129.283 , 128.8173 , 128.9062 , 128.2996 , 125.1422 , 126.7221 , 124.701] ,
  [10, 24.2    , 19.8    , 19.2    , 20.06   , 19.5    , 21.75   , 21.79   , 20.06   , 22.43   , 22.52    , 23.32    , 24.32    , 17.68    , 17.38    , 17.71]
];
PCA(M2);
