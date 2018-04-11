# abort on any error
set -e

DATAURL='https://storage.googleapis.com/comp551/processed.tar.gz'
GLOVEURL='https://nlp.stanford.edu/data/glove.6B.zip'

if [ ! -d .env ]; then
    echo "Creating virtualenv and downloading dependencies"
    virtualenv .env -p python3
    source .env/bin/activate
    pip install -r requirements.txt
else
    echo "Virtualenv already created"
fi

if [ ! -d data/processed ]; then
    DATAFNAME='.data.tmp.tar.gz'
    echo "Downloading processed dataset"
    curl -S $DATAURL > $DATAFNAME 
    tar -xf $DATAFNAME -C data
    rm $DATAFNAME
else
    echo "Dataset already downloaded"
fi

GLOVEDIR='resources/glove'
if [ ! -d $GLOVEDIR ]; then
    GLOVEFNAME='.glove.tmp.zip'
    echo "Downloading GloVe word embeddings"
    curl -S $GLOVEURL > $GLOVEFNAME
    mkdir $GLOVEDIR
    unzip $GLOVEFNAME -d $GLOVEDIR
    rm $GLOVEFNAME 
else
    echo "GloVe word embeddings already downloaded"
fi
