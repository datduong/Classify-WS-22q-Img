## Classify face images as WS, 22q11DS, other conditions, and unaffected. 

**[Paper](https://www.frontiersin.org/articles/10.3389/fgene.2022.864092/full)**

**[Dataset](https://figshare.com/articles/figure/ws-22q11-and-related-syndromes-04122023/22596217)** We make the versions of the publicly available images included in our analyses available for the purpose of reproducibility and research, with the assumption that these would not be used for purposes that would not be considered fair use. These data were compiled to produce a new, derivative work, which we offer as a whole. We cannot share the original images and related data with requestors, but have provided links to the URLs in the manuscript describing this work. We cannot guarantee that the URLs or images are accurate or up-to-date, and encourage interested parties to refer to the sources.

**Examples of individuals affected with [WS](https://en.wikipedia.org/wiki/Williams_syndrome) and [22q11DS](https://en.wikipedia.org/wiki/DiGeorge_syndrome).**

**Example surveys** for [WS](https://ncidccpssurveys.gov1.qualtrics.com/jfe/preview/SV_e9juDTdgWyFlcvY?Q_CHL=preview&Q_SurveyVersionID=current) and [22q11DS](https://ncidccpssurveys.gov1.qualtrics.com/jfe/preview/SV_d4EjJjIuNzjxBPw?Q_CHL=preview&Q_SurveyVersionID=current).

**Fake images were created from [our other github](https://github.com/datduong/stylegan2-ada-Ws-22q).**

## Dataset

Images can be kept in the same folder; our [train.csv and test.csv](https://github.com/datduong/Classify-WS-22q-Img/tree/master/Experiment/TrainTestCsv) are provided (you will need to change folder path based on your machine). 

We trained our images with [FairFace images](https://github.com/dchen236/FairFace) (these auxiliary images are included and formatted, so you don't need to download/format them).

## Train the classifier 

There are two types of classifier:
1. Classifier trained on real faces. 
2. Classifier trained on real + fake faces, which are
   - Type 1: Real + unique fake images (or at least, images are similar only by chance)
   - Type 2: Real + similar fake images (e.g., same hair styles/colors)
   - Type 3: Real + age transformation images (e.g., a fake person ages over time)
   - Type 4: Real + fake images of mixture of 2 diseases (e.g., a person with WS and 22q11DS)

Fake images were created based on [our github](https://github.com/datduong/stylegan2-ada-Ws-22q) redesigned from StyleGAN. The fake images are included in the dataset, so that that you don't have to train StyleGAN.

Training scripts are made and submitted to the server automatically using this python script (you may not want to do this if your school server has restrictions). Please change folder path to your own machines. We trained 5 fold cross-validation, and then combine these 5 models into a single classifier. [Example scripts for the 1st fold are provided here](https://github.com/datduong/Classify-WS-22q-Img/tree/master/Experiment/ExampleScripts). Ensemble script to combine all models is here; you will need to change model name and folder path with respect to your machine. 


Example of training. Please note, I'm using folder paths based on my own machines, [you will have to change the model paths](https://github.com/datduong/Classify-WS-22q-Img/blob/master/Demo/script-RealImg-0-11-29-15-25-10.sh#L28). 

[![asciicast](https://asciinema.org/a/452370.svg)](https://asciinema.org/a/452370)


Example of ensemble 5 fold cross-validation. Please note, I'm using folder paths based on my own machines, [you will have to change the model paths](https://github.com/datduong/Classify-WS-22q-Img/blob/master/ensemble.sh#L11). The last printed matrix is the confusion matrix. 

[![asciicast](https://asciinema.org/a/452373.svg)](https://asciinema.org/a/452373)

