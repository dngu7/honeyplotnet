module unload cuda/11.2
module load cuda/11.3.1
module load gcc/9.2.0
module load nano/latest
module load python/3.7.3
module load pdsh/latest

#FO DEEPSPEED
export CFLAGS="-I/data/csiro_od210966/envs/libraries/libaio-0.3.112/usr/include"
export LDFLAGS="-L/data/csiro_od210966/envs/libraries/libaio-0.3.112/usr/lib"

export DEFAULT_ENV=py37cu113t112 #py37cu113
export DATA_HOME=/data/csiro_od210966
export https_proxy=http://proxy.per.dug.com:3128
export http_proxy=http://proxy.per.dug.com:3128
set https_proxy=http://proxy.per.dug.com:3128
set http_proxy=http://proxy.per.dug.com:3128


source $DATA_HOME/envs/$DEFAULT_ENV/bin/activate

