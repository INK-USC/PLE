## PLE

Source code for SIGKDD'16 paper *[Label Noise Reduction in Entity Typing by Heterogeneous Partial-Label Embedding](http://xren7.web.engr.illinois.edu/kdd16-LNR.pdf)*. 

Given a text corpus with entity mentions *detected* and *heuristically labeled* by distant supervision, this code performs (1) *label noise reduction* over distant supervision, and (2) learning type classifiers over *de-noised* training data.

An end-to-end tool (corpus to typed entities) is under development. Please keep track of our updates.

## Dependencies

* python 2.7, g++

* [stanford coreNLP](http://stanfordnlp.github.io/CoreNLP/) and its [python wrapper](https://github.com/stanfordnlp/stanza).Place the library under `PLE/DataProcessor/`.

```
$ cd DataProcessor/
$ sudo pip install pexpect unidecode
$ git clone git@github.com:stanfordnlp/stanza.git
$ cd stanza
$ pip install -e .
$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
$ unzip stanford-corenlp-full-2016-10-31.zip
```
* [eigen 3.2.5](http://bitbucket.org/eigen/eigen/get/3.2.5.tar.bz2) (already included). 


## Data

We pre-processed three public datasets (train/test sets) to our JSON format. We ran [Stanford NER](https://nlp.stanford.edu/software/CRF-NER.shtml) on training set to detect entity mentions, and performed distant supervision using [DBpediaSpotlight](https://github.com/dbpedia-spotlight/dbpedia-spotlight) to assign type labels:
   * **Wiki** ([Ling & Weld, 2012](http://xiaoling.github.io/pubs/ling-aaai12.pdf)): 1.5M sentences sampled from 780k Wikipedia articles. 434 news sentences are manually annotated for evaluation. 113 entity types are organized into a 2-level hierarchy ([download JSON](https://drive.google.com/file/d/0B2ke42d0kYFfVC1fazdKYnVhYWs/view?usp=sharing))
   * **OntoNotes** ([Weischedel et al., 2011](https://catalog.ldc.upenn.edu/ldc2013t19)): 13k news articles with 77 of them are manually labeled for evaluation. 89 entity types are organized into a 3-level hierarchy. ([download JSON](https://drive.google.com/file/d/0B2ke42d0kYFfN1ZSVExLNlYwX1E/view?usp=sharing))
   * **BBN** ([Weischedel et al., 2005](https://catalog.ldc.upenn.edu/ldc2005t33)): 2,311 WSJ articles that are manually annotated using 93 types in a 2-level hierarchy. ([download JSON](https://drive.google.com/file/d/0B2ke42d0kYFfTEs0RGpuanRLQlE/view?usp=sharing))

- `Type hierarches` for each dataset are included.
- Please put the data files in the corresponding subdirectories under `PLE/Data/`.


## Default Run
Run PLE for the task of Reduce Label Noise on the BBN dataset

```
$ java -mx4g -cp "DataProcessor/stanford-corenlp-full-2016-10-31/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
$ ./run.sh  
```

## Makefile
If you need to re-compile `hple.cpp` under your own g++ environment
```
$ cd PLE/Model/ple/; make
```

## Parameters - run.sh
Dataset to run on.
```
Data="BBN"
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
