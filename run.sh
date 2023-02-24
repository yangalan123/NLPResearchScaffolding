project_dir=path/to/your/project

# create local environment
cd $project_dir
conda create -p ./env python=3.7
conda activate ./env

# fancy logging package
pip install loguru

# install pytorch with cuda
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# install huggingface
conda install transformers datasets evaluate
# large language model related utils, st=sentencepiece is useful for tokenizer loading
conda install sentencepiece
# cannot use conda, protobuf is set to be 3.20 as it is more compatible with released LLM so far
pip install protobuf==3.20
pip install accelerate
# setup huggingface cache dir
echo "export HF_HOME=${project_dir}/transformers" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=${project_dir}/transformers" >> ~/.bashrc

# install nltk, spacy, ..., often useful for pre-processing and post-processing
conda install nltk
python -m nltk.downloader all

pip install -U spacy
python -m spacy download en_core_web_sm

# scipy, sklearn..
conda install scipy scikit-learn

# install rouge for summary evaluation
pip install rouge-score

# setup vim
cd ~/
wget https://raw.githubusercontent.com/amix/vimrc/master/vimrcs/basic.vim
mv basic.vim .vimrc


# setup openai interface for using GPT-3/3.5/...
pip install openai

cd -

