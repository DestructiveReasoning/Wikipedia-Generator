# abort on any error
set -e

DATAURL='https://storage.googleapis.com/comp551/cnn_dm_data.zip'

if [ ! -f data/train.bin ]; then
    echo "Downloading dataset"
    curl -S $DATAURL > .data.tmp.zip
    unzip -j .data.tmp.zip -d data
    rm .data.tmp.zip
else
    echo "Dataset already downloaded"
fi
