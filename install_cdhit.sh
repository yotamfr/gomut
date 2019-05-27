#!/bin/sh
wget -nc http://github.com/weizhongli/cdhit/releases/download/V4.8.1/cd-hit-v4.8.1-2019-0228.tar.gz
tar xvf cd-hit-v4.8.1-2019-0228.tar.gz --gunzip
cd cd-hit-v4.8.1-2019-0228
cd cd-hit-auxtools
make
cd ..
make
