# ISAC-Integrated-Sensing-And-Communications
### Paper: Channel Coding meets Sequence Design via Machine Learning for Integrated Sensing and Communications
### Caution: Didn't recover original performance

## Setup
You can run on your own environment(less library version limitation).  
Below commands are just recommendation.
```bash
$ conda create env --name ccsd
$ conda activate ccsd
$ pip install -r requirements.txt
```
## Train (24M samples)
lambda==0.9 means learning sensing technic(ACSL).  
lambda==0 means only to learn error correction.   

ae is autoencoder(original paper).  
tf is transformer autoencoder(proposal method).  
```bash
$ python3 main.py --model ae --lambda 0.9
$ python3 main.py --model ae --lambda 0
$ python3 main.py --model tf --lambda 0.9
$ python3 main.py --model tf --lambda 0
```
## Plot (1M samples)
Run these codes then you can see the results in the figures folder.  
Transformer autoencoder takes several minutes to plot outputs.  
BER can be -∞ dB because of no error in message vectors.
```bash
$ python3 load_and_plot.py 
```
### Notice 
Pre-trained models are provided.  
Set "batch_first=True" to improve inference performance for transformer.  
Folder named 'history' is losses of models.  
## Reference
Original paper:  
&emsp;arXiv:2503.23119 [eess.SP]  
&emsp;(or arXiv:2503.23119v1 [eess.SP] for this version)  
&emsp;https://doi.org/10.48550/arXiv.2503.23119
