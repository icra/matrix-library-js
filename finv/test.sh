echo "
  import {finv} from './finv.js';
  finv(0.95 , 5        , 10);
  finv(0.95 , 10       , 40);
  finv(0.95 , Infinity , Infinity);
  finv(0.95 , 120      , Infinity);
  finv(0.95 , Infinity , 120);
  finv(0.95 , 10       , 31);
  finv(0.95 , 11       , 4);
  finv(0.95 , 120      , 5e5);
  finv(0.95 , 5e5      , 120);
  finv(0.95 , 121      , 121);
  finv(0.95 , 150      , 150);
  finv(0.95 , 200      , 200);
  finv(0.95 , 300      , 300);
  finv(0.95 , 500      , 500);
  finv(0.95 , 1000     , 1000);
  finv(0.95 , 5000     , 5000);
  finv(0.95 , 1e4      , 1e4);
  finv(0.95 , 1e5      , 1e5);
  finv(0.95 , 2e5      , 2e5);
  finv(0.95 , 232101   , 232110);
  finv(0.95 , 3e5      , 3e5);
  finv(0.95 , 4e5      , 4e5);
  finv(0.95 , 5e5      , 5e5);
" > finv.tests.js

echo "
  qf(0.95 , 5        , 10);
  qf(0.95 , 10       , 40);
  qf(0.95 , Inf , Inf);
  qf(0.95 , 120      , Inf);
  qf(0.95 , Inf , 120);
  qf(0.95 , 10       , 31);
  qf(0.95 , 11       , 4);
  qf(0.95 , 120      , 5e5);
  qf(0.95 , 5e5      , 120);
  qf(0.95 , 121      , 121);
  qf(0.95 , 150      , 150);
  qf(0.95 , 200      , 200);
  qf(0.95 , 300      , 300);
  qf(0.95 , 500      , 500);
  qf(0.95 , 1000     , 1000);
  qf(0.95 , 5000     , 5000);
  qf(0.95 , 1e4      , 1e4);
  qf(0.95 , 1e5      , 1e5);
  qf(0.95 , 2e5      , 2e5);
  qf(0.95 , 232101   , 232110);
  qf(0.95 , 3e5      , 3e5);
  qf(0.95 , 4e5      , 4e5);
  qf(0.95 , 5e5      , 5e5);
" > finv.tests.r

node    finv.tests.js > finv.tests.js.txt
Rscript finv.tests.r  > finv.tests.r.txt

paste finv.tests.js.txt finv.tests.r.txt | column -t

rm finv.tests.*
