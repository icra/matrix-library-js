v1  = 3e5;
v2  = 3e5;
inc = 1;
TOL = 1e-8;
while(T){
  f1  = qf(0.95,v1,v2)
  v1  = v1+inc;
  v2  = v2+inc;
  f2  = qf(0.95,v1,v2)
  v1  = v1+inc;
  v2  = v2+inc;
  dif = abs(f1-f2);
  cat(v1,v2,f1,f2,dif,'\n')
  if(dif<TOL) break;
}
