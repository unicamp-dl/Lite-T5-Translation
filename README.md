# Lite PT-EN-Translator

This repository brings an implementation of T5 for translation in PT-EN and EN-PT tasks using a modest hardware setup. We propose some changes in tokenizator and post-processing that improves the result and used a Portuguese pretrained model for the translation. We have here also a link to the arxiv paper and the best reported weights for wmt submission and ParaCrawl. The training process for T5 in portuguese works like Figure 1. 

<img src="https://github.com/unicamp-dl/Lite-T5-Translation/blob/master/figs/t5.png" width="500">
Figure 1: T5 training strategy with adition of red conections.  The purple boxeswere the fine-tuning part performed by this work.  Created using figure from [1]

For the first part of the project, we used https://paracrawl.eu/ corpus. We trained using 5M+ data from ParaCrawl. We just did not use more data because it was taking too much time. In theory, it would improve the results.

After step one, we also fine-tuned in a 6M+ Portuguese-English corpora of scientifical data. We did not evaluate these results with ParaCrawl. Feel free to do so and compare the results. This translator looks better than the pracrawl one, specially for Brazilian Portuguese sentences. Therefore, don't forget to test both translators to your project. 

Another important contribution of this repository is the creation a corpus called ParaCrawl 99k. The ParaCrawl 99k is composed of two small corpus of Paracrawl containing the Google Translate En - PT-pt and Google Translate PT-pt - En translations of 99k sentences. This costed around $300 each to train, so you can save some money if you want to compare your results with Google Translate (GT). Pay attention to remove such itens from ParaCrawl before running it. All descriptions to this dataset, best practices and uses are in the Readme file of ParaCrawl99k folder. 

## Instalation and Usage

We used Python 3.8 for the experiments. Probably 3.7 should be fine too, but I cannot garantee any other version. I sugest using Python 3.8.

To install it, just run 'pip install -r requirements.txt'

It you want to train it, make sure you have a 8GB GPU or change the batch sizes and train it again. On CPU it should work well for evaluating, but it must be time consuming for training. I did not try it.

## Organization

Here we have code for reproducing our results (training and testing) and for using it. Modules and notebooks are available for testing.

| Component | Description |
| ------ | ------ |
| [pt_en_translator](https://github.com/unicamp-dl/Lite-T5-Translation/tree/master/pt_en_translator/) | Contains all codes for running the training, testing and just using the module to your own projects.
| [paracrawl99k](https://github.com/unicamp-dl/Lite-T5-Translation/tree/master/ParaCrawl99k/) | Contains pickle of Google Translate results for a subset of ParaCrawl. Contains also notebooks to help dividing ParaCrawl in the 20M and 99k datasets|
| [models](https://github.com/unicamp-dl/Lite-T5-Translation/tree/master/models/) | Contains the link for the models of translation generated by our team. |
| [reports](https://github.com/unicamp-dl/Lite-T5-Translation/tree/master/reports/) | Contains the pdf of any report or publication created from this project. |
| [notebooks](https://github.com/unicamp-dl/Lite-T5-Translation/tree/master/notebooks/) | Examples on how to generate and use our project. |

## Training on your own data

If you are going to train using our model or use our testing corpora, It is important to deduplicate your data with testing set provided by us. Also, we think it is suitable to run MinHash LSH algorithm to discard any sentence with > 0.7 Jaccard Similarity with any testing word. We have code for this in paracrawl99k and an example in the Notebooks folder.

## Results

Here we are going to compare the results of our algoritm with GT with the translated subset of ParaCrawl and in WMT19' Biomedical Pt-En and En-Pt tasks, comparing our results with the winning algorithm and MarianMT Implementation from https://huggingface.co/transformers/model_doc/marian.html. 

The results comparing with GT in the subset of ParaCrawl are available in Table 1. The results comparing with WMT19' Test Set is available in Table 2. It is important to notice that we obtain SOTA results in PT-EN abd EN-PT translation in WMT19'. It is also important to notice that our project was trained in Pt-pt (ParaCrawl) and as WMT19's weights are Pt-br, therefore the results will change a little bit if you are going to use in other datasets. 

We also participated in the WMT'20 Biomedical challenge, but without our best solutions (we improved en-pt in +9.75 and +2.27 BLEU scores in pt-en). We are waiting for the official data to compare the submissions with our best one.

Table 1: Results in ParaCrawl subset with sacreBLEU score

![alt text](https://github.com/unicamp-dl/Lite-T5-Translation/blob/master/figs/results_gt_2.png)


Table 2: Results in WMT19' test set with BLEU score

![alt text](https://github.com/unicamp-dl/Lite-T5-Translation/blob/master/figs/results_wmt.png)

About the quality of the data for training, we understand that there is a lot to improve. But to get a sense of the translator's quality, you're reading a text excerpt translated by him. The translator is able to generate translations in a more complex context with difficult words. All this last paragraph is composed of text automatically translated by our project.

## Future Work

We want to pre-train data in portuguese and english dataset together using a new trained in both languages SentensePiece tokenizer. We believe this could improve the results. We also believe that using larger models would increase the performance of the work.

If you want to contribute to this work or team up with me for some other task (or other challenge in Translation), let me know! Feel free to create issues and share your comments!


## Citation

For now, we have a paper you can cite in arxiv:
```bibtex
@inproceedings{lopes-etal-2020-lite,
    title = "Lite Training Strategies for {P}ortuguese-{E}nglish and {E}nglish-{P}ortuguese Translation",
    author = "Lopes, Alexandre  and
      Nogueira, Rodrigo  and
      Lotufo, Roberto  and
      Pedrini, Helio",
    booktitle = "Proceedings of the Fifth Conference on Machine Translation",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.wmt-1.90",
    pages = "833--840",
}
```
## References

[1] Colin  Raffel,   Noam  Shazeer,   Adam  Roberts,   Katherine  Lee,   SharanNarang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu.  Exploring the limits of transfer learning with a unified text-to-text transformer.arXivpreprint arXiv:1910.10683, 2019
