## Heterogeneous Partial-Label Embedding

Source code and data for SIGKDD'16 paper *[Label Noise Reduction in Entity Typing by Heterogeneous Partial-Label Embedding](http://xren7.web.engr.illinois.edu/kdd16-LNR.pdf)*. 

Given a text corpus with entity mentions *detected* and *heuristically labeled* by distant supervision, this code performs (1) *label noise reduction* over distant supervision, and (2) learning type classifiers over *de-noised* training data.

An end-to-end tool (corpus to typed entities) is under development. Please keep track of our updates.

## Dependencies

* python 2.7, g++
* Python library dependencies
```
$ pip install pexpect unidecode six requests protobuf
```
* Setup [stanford coreNLP](http://stanfordnlp.github.io/CoreNLP/) and its [python wrapper](https://github.com/stanfordnlp/stanza).
```
$ cd DataProcessor/
$ git clone git@github.com:stanfordnlp/stanza.git
$ cd stanza
$ pip install -e .
$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
$ unzip stanford-corenlp-full-2016-10-31.zip
$ rm stanford-corenlp-full-2016-10-31.zip
```
* [eigen 3.2.5](http://bitbucket.org/eigen/eigen/get/3.2.5.tar.bz2) (already included). 


## Data

We pre-processed three public datasets (train/test sets) to our JSON format. We ran [Stanford NER](https://nlp.stanford.edu/software/CRF-NER.shtml) on training set to detect entity mentions, and performed distant supervision using [DBpediaSpotlight](https://github.com/dbpedia-spotlight/dbpedia-spotlight) to assign type labels:
   * **Wiki** ([Ling & Weld, 2012](http://xiaoling.github.io/pubs/ling-aaai12.pdf)): 1.5M sentences sampled from 780k Wikipedia articles. 434 news sentences are manually annotated for evaluation. 113 entity types are organized into a 2-level hierarchy ([download JSON](https://drive.google.com/file/d/0B2ke42d0kYFfVC1fazdKYnVhYWs/view?usp=sharing))
   * **OntoNotes** ([Weischedel et al., 2011](https://catalog.ldc.upenn.edu/ldc2013t19)): 13k news articles with 77 of them are manually labeled for evaluation. 89 entity types are organized into a 3-level hierarchy. ([download JSON](https://drive.google.com/file/d/0B2ke42d0kYFfN1ZSVExLNlYwX1E/view?usp=sharing))
   * **BBN** ([Weischedel et al., 2005](https://catalog.ldc.upenn.edu/ldc2005t33)): 2,311 WSJ articles that are manually annotated using 93 types in a 2-level hierarchy. ([download JSON](https://drive.google.com/file/d/0B2ke42d0kYFfTEs0RGpuanRLQlE/view?usp=sharing))

- `Type hierarches` for each dataset are included.
- Please put the data files in the corresponding subdirectories under `PLE/Data/`.


## System Output
The output on [BBN dataset](https://drive.google.com/file/d/0B2ke42d0kYFfTEs0RGpuanRLQlE/view?usp=sharing) can be found [here](https://raw.githubusercontent.com/shanzhenren/PLE/master/Results/BBN/predictionInText_hple_hete_feature_perceptron.txt). Each line is a sentence in the test data of BBN, with entity mentions and their fine-grained entity typed identified.


## Makefile
We have included compilied binaries. If you need to re-compile `hple.cpp` under your own g++ environment
```
$ cd PLE/Model/ple/; make
```

## Default Run
Run PLE for the task of Reduce Label Noise on the BBN dataset

```
$ java -mx4g -cp "DataProcessor/stanford-corenlp-full-2016-10-31/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
$ ./run.sh  
```
- The [run.sh](https://github.com/shanzhenren/PLE/blob/master/run.sh) contains parameters for running on three datasets.


## Parameters - run.sh
Dataset to run on.
```
Data="BBN"
```


## Evaluation
Evaluate prediction results (by classifier trained on de-noised data) over test data
```
python Evaluation/evaluation.py -DATA(BBN/ontonotes/FIGER) -METHOD(hple/...) -EMB_MODE(hete_feature)
```


## Reference
Please cite the following paper if you found the codes/datasets useful:
```
@inproceedings{ren2016label,
  title={Label Noise Reduction in Entity Typing by Heterogeneous Partial-Label Embedding},
  author={Ren, Xiang and He, Wenqi and Qu, Meng and Voss, Clare R and Ji, Heng and Han, Jiawei},
  booktitle={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={1825--1834},
  year={2016},
  organization={ACM}
}
```
