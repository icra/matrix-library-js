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
  let A=[[1,2],[3,4]];
  let B=[[1,2],[3,4]];
  let C=[[3,4],[5,6]];
  assert(are_equal(A,B)==true, 'A and B are equal')
  assert(are_equal(A,C)==false,'A and C are different');
}

//transposed matrix
{
  let A = transposed([[1,2,3],[4,5,6],[7,8,9]]);
  let B =            [[1,4,7],[2,5,8],[3,6,9]];
  assert(are_equal(A,B),'B is transposed of A');
}

//minors
{
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
  let A=[[1,4,7],[2,5,8],[3,6,9]];
  let AA=multiply(A,A);
  assert(are_equal(AA,[[30,66,102],[36,81,126],[42,96,150]]),'Error in matrix multiplication')
}

//matrix multiplied by scalar
{
  let A=[[1,2,3],[4,5,6],[7,8,9]];
  let B=escalate(A,2);
  assert(are_equal(B,[[2,4,6],[8,10,12],[14,16,18]]),'2A is B');
}

//matrix sum
{
  let A=[[1,2,3],[4,5,6],[7,8,9]];
  let B=[[1,1,1],[2,2,2],[3,3,3]];
  let C=sum(A,B);
  assert(are_equal(C,[[2,3,4],[6,7,8],[10,11,12]]),'A+B is C');
}

//matrix subtraction
{
  let A=[[1,4,7],[2,5,8],[3,6,9]];
  let B=[[1,4,7],[1,4,7],[1,4,7]];
  let C=subtract(A,B);
  assert(are_equal(C,[[0,0,0],[1,1,1],[2,2,2]]),'A-B is C');
}

//determinants
{
  assert(determinant([[1,3],[2,4]]                                  )== -2,"Determinant should be -2" );
  assert(determinant([[1,4,7],[2,5,8],[3,6,9]]                      )==  0,"Determinant should be 0"  );
  assert(determinant([[1,2,-1,3],[0,-3,2,2],[3,-2,1,5],[-3,3,2,0]]  )==-80,"Determinant should be -80");
  assert(determinant([[2,4,-2,4],[1,5,5,11],[-1,-3,-2,-4],[2,6,6,8]])==-12,"Determinant should be -12");
}

//adjoint matrix
{
  let A=[[1,0,4],[0,3,0],[2,0,5]];
  let B=adjoint(A);
  assert(are_equal(B,[[15,0,-6],[0,-3,0],[-12,0,3]]),"B is the adjoint of A");
}

//inverse matrix
{
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

console.log("All tests passed");
