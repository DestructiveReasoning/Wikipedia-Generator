# abort on any error
set -e

DATAURL='https://storage.googleapis.com/comp551/processed.tar.gz'
GLOVEURL='https://nlp.stanford.edu/data/glove.6B.zip'

if [ ! -d .env ]; then
    echo "Creating virtualenv and downloading dependencies"
    virtualenv .env
    source .env/bin/activate
    pip install -r requirements.txt
else
    echo "Virtualenv already created"
fi

if [ ! -d data/processed ]; then
    echo "Downloading processed dataset"
    curl -S $DATAURL > .data.tmp.zip
    unzip .data.tmp.zip -d data
    rm .data.tmp.zip
else
    echo "Dataset already downloaded"
fi

if [ ! -d resources/glove.6B ]; then
    echo "Downloading GloVe word embeddings"
    curl -S $GLOVEURL > .glove.tmp.zip
    unzip .glove.tmp.zip -d resources
    rm .glove.tmp.zip
else
    echo "GloVe word embeddings already downloaded"
fi
