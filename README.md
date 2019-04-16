# PCNN+ONE based Relation Extraction via Tensorflow Framework

This is a tf1.1.3 based repo which use PCNN+ONE to solve Relation Extraction problems. Most of the ideas borrow from [OpenNRE](https://github.com/thunlp/OpenNRE).

Well, I have to say, Keras is a great framework and easy to use in model stacking. But, it is not enough flexible to some self-design and research project. Maybe it is because I have not got the spirit of Keras... So I decide to use tensorflow instead of keras in this repo. You may see the former commits contains keras  code. But I have changed the code into tf already in the last commit.

This repo is not support for gpu yet, but I'm working on it, since I don't have a gpu yet...

I am a newbie and still learning, so feel free to raise some issues and make pull requests.

## How to Use

1. download the data and organize data in a proper file dir structure
2. `python singlefile.py`
3. `tensorboard --logdir ./summary/pcnn_one_train_demo/`

## Dataset

### NYT10

> NYT10 is a distantly supervised dataset originally released by the paper "Sebastian Riedel, Limin Yao, and Andrew McCallum. Modeling relations and their mentions without labeled text.". Here is the download [link](http://iesl.cs.umass.edu/riedel/ecml/) for the original data.

> We've provided a toolkit to convert the original NYT10 data into JSON format that `OpenNRE` could use. You could download the original data + toolkit from [Google Drive](https://drive.google.com/file/d/1eSGYObt-SRLccvYCsWaHx1ldurp9eDN_/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/11391e48b72749d8b60a/?dl=1). Further instructions are included in the toolkit.

## References

### Papers

- Zeng et al. Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Network. 2015
- Zeng et al. Relation Classification via Convolutional Deep Neural Network. 2014

### Reading Materials

- [Relation Extraction Note](http://shomy.top/2018/02/28/relation-extraction/)
- [Differences Between PCNN and PCNN+ONE](https://github.com/ShomyLiu/pytorch-relation-extraction/issues/10)
- [Keras Data Generator](https://blog.csdn.net/m0_37477175/article/details/79716312)
- [MIL loss](https://github.com/keras-team/keras/issues/3415)
- [Keras Text Classification by using Pretrained Word Vectors](https://www.jianshu.com/p/7eed068ff353)
- [Using Pre-Trained Word Embeddings in a Keras Model](https://kiseliu.github.io/2016/08/03/using-pre-trained-word-embeddings-in-a-keras-model/)

### Codes

- [ShomyLiu/pytorch-relation-extraction](https://github.com/ShomyLiu/pytorch-relation-extraction)
- [smilelhh/ds_pcnns](https://github.com/smilelhh/ds_pcnns)
- [dancsalo/TensorFlow-MIL](https://github.com/dancsalo/TensorFlow-MIL)
- [thunlp/OpenNRE](https://github.com/thunlp/OpenNRE)
- [ShulinCao/OpenNRE-PyTorch](https://github.com/ShulinCao/OpenNRE-PyTorch)