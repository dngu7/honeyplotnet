export DATA_HOME=/media/dingdong/DATA3

#For deepspeed
export CFLAGS="-I$DATA_HOME/envs/libraries/libaio-0.3.112/usr/include"
export LDFLAGS="-L$DATA_HOME/envs/libraries/libaio-0.3.112/usr/lib"

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

export DEFAULT_ENV=py37cu116t112
source $DATA_HOME/envs/$DEFAULT_ENV/bin/activate


