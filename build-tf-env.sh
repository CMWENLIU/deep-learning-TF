cd ~
# You can change what anaconda version you want at 
# https://repo.continuum.io/archive/
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh -b -p ~/anaconda
rm Anaconda3-5.0.1-Linux-x86_64.sh
echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashrc 

# Refresh basically
source ~/.bashrc
 
conda update conda
# Refresh for updating
source ~/.bashrc
 
# Install tf-gpu v1.3
#conda install -c jjhelmus tensorflow-gpu-base 
# Install tensorflow
conda install tensorflow=1.3
#install gensim
conda install gensim
 
source ~/.bashrc
  
cd

