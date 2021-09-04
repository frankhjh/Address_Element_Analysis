# Chinese Address Element Parsing

## Introduction
The aim of this task is to transform the address sequence into label sequence, each element of the address corresponds to one label. There are 21 different kinds of tags, for example, province(prov),city,district and so on. Besides, the task uses the **BIEO** system, for example, 'B' represents the beginning of one element,'E' represents the end of one element.

For a specific location of one address, a complete label will be something like 'B-prov'.

## Data
The train data and dev data are stored in form of ***.conll***, and the test file is stored in form of ***.txt***.the size of test data is 50000.

## Model
for this task, I use the **LSTM+CRF** model, which is quite popular in dealing with this kind of **NER**(named-entity recognition.) problem.

The key idea is that we use LSTM to extract feature, and use CRF to learn the relationship between different labels. In other words, the aim of CRF is to set ristrictions of labeling transfering. For example, it is not possible that the 'B-prov' follows with 'E-prov'.

## Run the model
you can run the model by simply run the following command
`python main.py --epochs 20 --learning_rate 1e-2 --device 'cpu'`

Of course,you can change the parameters as you like. What'more,if you want to change the model parameters, you can further check the `lstm_crf.py`.

## Output
The final output after you run the code will be save in `./output/final_out.txt`,I also save the intermediate result in file `./output/tmp_predict.json`,if you are interested, you could check it too.

## Some references
[Very good introduction about LSTM-CRF model](https://www.cnblogs.com/createMoMo/p/7529885.html)

[Pytorch Official Tutorial](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#bi-lstm-conditional-random-field-discussion)

[LSTM-CRF Faster Version](https://github.com/mali19064/LSTM-CRF-pytorch-faster)




