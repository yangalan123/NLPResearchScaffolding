project_dir=path/to/your/project

conda create -p ./env python=3.7
conda activate ./env

pip install loguru

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

conda install transformers nltk scikit-learn 

cd ~/
wget https://raw.githubusercontent.com/amix/vimrc/master/vimrcs/basic.vim

mv basic.vim .vimrc

echo "export HF_HOME=${project_dir}/transformers" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=${project_dir}/transformers" >> ~/.bashrc

cd -

