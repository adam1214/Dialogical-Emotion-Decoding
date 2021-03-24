# Dialogical Emotion Decoding

This is an implementation of Dialogical Emotion Decoder presented in this year ICASSP 2020. In this repo, we use IEMOCAP as an example to 
evaluate the effectiveness of DED.

## Overview
	
<p align="center">
  <img src="img/ded.png" width="500" height="400">
</p>


## Note
+ The performance is **better** than shown in the paper because I found a little bug in the rescoring part.
+ To test on your own emotion classifier, replace `data/outputs.pkl` with your own outputs.
	+ Dict, {utt_id: logit} where utt_id is the utterance name in [IEMOCAP](https://sail.usc.edu/iemocap/release_form.php).

## Requirements

```bash
pip3 install virtualenv
virtualenv --python=python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Dataset

Currently this repo only supports [IEMOCAP](https://sail.usc.edu/iemocap/release_form.php).

## Arguments

The definitions of the args are described in `ded/arguments.py`. You can modify all args there.

## Usage

```bash
python3 main.py --verbosity 1 --result_file RESULT_FILE
```


## Results
Results of DED with beam size = 5

| Model |  Original Training Data UAR  | Original Training Data ACC  |Class to Class Training Data UAR  | Class to Class Training Data ACC  |Utt to Utt Training Data UAR  | Utt to Utt Training Data ACC  |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
| Pretrained Classifier |0.671|0.653|-|-|-|-|
| Original DED |0.710|0.695|-|-|-|-|
| Bigram-Sequential-Softmax DED      |0.677|0.659|0.670|0.654|0.674|0.655|
| Bigram-Sequential-Nonsoftmax DED  |0.687|0.669|0.670|0.665|0.684|0.666|
| Bigram-Intra-Softmax DED      |0.683|0.665|0.682|0.666|0.682|0.664|
| Bigram-Intra-Nonsoftmax DED  |0.718|0.704|0.691|0.686|0.710|0.696|
| Trigram-Sequential-Softmax DED     |0.676|0.657|0.675|0.658|0.674|0.655|
| Trigram-Sequential-Nonsoftmax DED |0.697|0.682|0.680|0.673|0.700|0.684|
| Trigram-Intra-Softmax DED     |||||||
| Trigram-Intra-Nonsoftmax DED |||||||
| Trigram-Sequential-Softmax-Add-1-Smooth DED |0.676|0.657|0.675|0.658|0.674|0.655|
| Trigram-Sequential-Nonsoftmax-Add-1-Smooth DED |0.700|0.685|0.682|0.675|0.701|0.684|
| Trigram-Intra-Softmax-Add-1-Smooth DED |||||||
| Trigram-Intra-Nonsoftmax-Add-1-Smooth DED |||||||

## Oral Presentation
[![IMAGE ALT TEXT](img/ICASSP20.png)](https://www.youtube.com/watch?v=Ti4foNyrvzo)