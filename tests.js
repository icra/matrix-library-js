/*
  Tests for Matrix Library in Javascript

  Tests include:
    - checks for matrix objects
    - checks for hardcoded numeric examples
*/
import {
  assert,
  zeros,
  ones,
  check_matrix,
  are_equal,
  transposed,
  multiply,
  minor,
  escalate,
  sum,
  subtract,
  determinant,
  adjoint,
  inverse,
  gaussian_elimination,
  covariance_matrix,
  PCA,
} from './module.js';

/*
  Check that something supposed to fail does fail
*/
function test_must_fail(fx){
  //fx: function callback
  let failed=false;
  try{
    fx();
  }catch(e){
    failed=true;
    //console.log(e);
  }finally{
    if(failed==false){
      throw("This test is supposed to fail");
    }
  }
}

/*
  Tests
*/
assert(1==1,"1 is 1");
test_must_fail(()=>{assert(0==1,"0 is not 1");});

assert(zeros(5).length==5);
assert(ones(5).length==5);
check_matrix([[1,2],[3,4]]);

//check malformed matrices
test_must_fail(()=>{check_matrix("A"           )}); //string
test_must_fail(()=>{check_matrix([]            )}); //empty array
test_must_fail(()=>{check_matrix([1,2,3]       )}); //simple array (no columns)
test_must_fail(()=>{check_matrix([[],[1],[1]]  )}); //0 elements in one column
test_must_fail(()=>{check_matrix([[1],[],[1,2]])}); //different number of elements in columns

//equal matrices
{
  console.log("Testing matrix equality")
  let A=[[1,2],[3,4]];
  let B=[[1,2],[3,4]];
  let C=[[3,4],[5,6]];
  assert(are_equal(A,B)==true, 'A and B are equal')
  assert(are_equal(A,C)==false,'A and C are different');
}

//transposed matrix
{
  console.log("Testing transposed matrix")
  let A = transposed([[1,2,3],[4,5,6],[7,8,9]]);
  let B =            [[1,4,7],[2,5,8],[3,6,9]];
  assert(are_equal(A,B),'B is transposed of A');
}

//minors
{
  console.log("Testing matrix minor")
  let A=[
    [1,4,7],
    [2,5,8],
    [3,6,9],
  ];
  let Am00 = minor(A,0,0);
  assert(are_equal(Am00,[[5,8],[6,9]]),'A minor i=0, j=0');
  let Am11 = minor(A,1,1);
  assert(are_equal(Am11,[[1,7],[3,9]]),'A minor i=1, j=1');
}

//matrix multiplication
{
  console.log("Testing matrix multiplication")
  let A=[[1,4,7],[2,5,8],[3,6,9]];
  let AA=multiply(A,A);
  assert(are_equal(AA,[[30,66,102],[36,81,126],[42,96,150]]),'Error in matrix multiplication')

  let M=[[5,2,-3]];
  assert(multiply(transposed(M),M)[0].length==1,"wrong lengths");
  assert(multiply(M,transposed(M))[0].length==3,"wrong lengths");
}

//matrix multiplied by scalar
{
  console.log("Testing matrix times scalar")
  let A=[[1,2,3],[4,5,6],[7,8,9]];
  let B=escalate(A,2);
  assert(are_equal(B,[[2,4,6],[8,10,12],[14,16,18]]),'2A is B');
}

//matrix sum
{
  console.log("Testing matrix sum")
  let A=[[1,2,3],[4,5,6],[7,8,9]];
  let B=[[1,1,1],[2,2,2],[3,3,3]];
  let C=sum(A,B);
  assert(are_equal(C,[[2,3,4],[6,7,8],[10,11,12]]),'A+B is C');
}

//matrix subtraction
{
  console.log("Testing matrix subtraction")
  let A=[[1,4,7],[2,5,8],[3,6,9]];
  let B=[[1,4,7],[1,4,7],[1,4,7]];
  let C=subtract(A,B);
  assert(are_equal(C,[[0,0,0],[1,1,1],[2,2,2]]),'A-B is C');
}

//determinants
{
  console.log("Testing determinant")
  assert(determinant([[1,3],[2,4]]                                    )== -2,"Determinant is not -2" );
  assert(determinant([[1,4,7],[2,5,8],[3,6,9]]                        )==  0,"Determinant is not 0"  );
  assert(determinant([[2,4,-2,4],[1,5,5,11],[-1,-3,-2,-4],[2,6,6,8]]  )==-12,"Determinant is not -12");

  //case where there is an error of 1e-13
  let d    = determinant([[0,0,0,-1],[-2,-6,11,-2],[-5,3,5,-1],[8,1,-3,3]]);
  let TOL  = 1e-10;
  let diff = Math.abs(-441-d);
  assert(diff<TOL,`-441 != ${d} (${diff})`)
}

//adjoint matrix
{
  console.log("Testing adjoint matrix")
  let A=[[1,0,4],[0,3,0],[2,0,5]];
  let B=adjoint(A);
  assert(are_equal(B,[[15,0,-6],[0,-3,0],[-12,0,3]]),"B is the adjoint of A");
}

//inverse matrix
{
  console.log("Testing inverse matrix")
  let A=[[4,0,1],[0,0,-2],[0,-2,8]];
  let Ai=inverse(A);
  let I=multiply(A,Ai);
  assert(are_equal(I,[[1,0,0],[0,1,0],[0,0,1]]),"Ai is the inverse of A");
}

//example from "module.js" header comment
{
  let A=[
    [1, 3, 1],
    [2, 4, 1],
    [1, 1, 1],
  ];
  let d = determinant(A);  //
  let T = transposed(A);   //
  let B = multiply(A,A);   //
  let C = escalate(A,5);   //
  let D = sum(A,B);        //
  let E = subtract(A,B);   //
  let F = minor(A,1,2);    //
  let G = adjoint(A);      //
  let H = inverse(A);      //
  let I = multiply(H,A);   //
  //console.log({d,T,B,C,D,E,F,G,H,I});
}

{//test for gaussian elimination
  console.log("Testing gaussian elimination")
  //create large matrix of random numbers with zeros
  let n=100;
  let M=[];
  for(let i=0;i<n;i++){
    let new_row=[];
    for(let j=0;j<n;j++){
      let r = Math.max(0,1000*Math.random()-500);
      r = r ? r-250:0;
      new_row.push(r);
    }
    M.push(new_row);
  }
  let T=gaussian_elimination(M);
  let detM=determinant(M);
  let detT=determinant(T);
  assert(detM==detT,"Determinants are different");
}

{//test for covariance matrix TODO
  console.log("Testing covariance matrix")
  let X =[[15,35,20,14,28],[12.5,15.8,9.3,20.1,5.2],[50,55,70,65,80]];
  let S = covariance_matrix(X);
  let Xb = [
    [115.25,115.91,115.05,116.21,115.90,115.55,114.98,115.25,116.15,115.92,115.75,114.90,116.01,115.83,115.29,115.63,115.47,115.58,115.72,115.40],
    [1.04,1.06,1.09,1.05,1.07,1.06,1.05,1.10,1.09,1.05,0.99,1.06,1.05,1.07,1.11,1.04,1.03,1.05,1.06,1.04],
  ];
  let Sb = covariance_matrix(Xb);
  //console.log(Sb);
}

{//tests for PCA
  console.log("Testing PCA");
  const M=[//Ariane flights dataset
    [121.9   , 120.9   , 124.1   , 122.2   , 119.8   , 125.745 , 123.43  , 123.883 , 125.353 , 126.0153 , 125.3861 , 124.4702 , 121.3318 , 123.4016 , 121.654] ,
    [91.137  , 94.685  , 90.322  , 90.025  , 86.933  , 89.343  , 89.125  , 94.878  , 89.303  , 93.7609  , 92.4001  , 90.0061  , 87.8128  , 88.6698  , 85.324]  ,
    [108.096 , 111.838 , 107.391 , 106.646 , 103.966 , 105.897 , 107.203 , 107.497 , 107.976  , 109.373 , 111.612  , 109.7599 , 105.9164 , 110.0603 , 110.419] ,
    [126.2   , 124.3   , 126.9   , 125.9   , 123.8   , 129.197 , 129.044 , 126.674 , 129.283 , 128.8173 , 128.9062 , 128.2996 , 125.1422 , 126.7221 , 124.701] ,
    [24.2    , 19.8    , 19.2    , 20.06   , 19.5    , 21.75   , 21.79   , 20.06   , 22.43   , 22.52    , 23.32    , 24.32    , 17.68    , 17.38    , 17.71]
  ];
  let result = PCA(M);
  //console.log(result);
}

console.log("\nALL TESTS PASSED");
