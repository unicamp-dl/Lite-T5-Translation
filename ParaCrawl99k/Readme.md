## ParaCrawl99k

ParaCrawl99k is a dataset that allows you to comparing your result to Google Translate (GT). It is fully available to allow you to compare your translations with 
Google Translate. Please note that we used Pt-pt translation since ParaCrawl contains mostly pt-pt translations. There are files containing 120k sentences and 99k
The 120k goes to 99k because we removed data from testing that had a jaccard similarity with any sentence in training larger than 0.7. Therefore, you can use the rest
of the 20M-120k data to train your dataset. Please, even if using the 99k version, do not include any of the 120k in your training data. 

Google Translate metrics are available in the paper. You can also calculate yours, since there are two pickle files. Each pickle file contains the Original -> GT version.
It is basically a single parallel corpus of two languages but with four results (2 GT and 2 originals, each one in a mixed pair). For instance, GT_Pred_EnPt.pkl has
original English and Portuguese GT and GT_Pred_PtEn.pkl has original text in Portuguese with English GT translation.


## License

We follow ParaCrawl license scheme. Therefore, these data are released under this licensing scheme:

- We do not own any of the text from which these data has been extracted.
- We license the actual packaging of these parallel data under the Creative Commons CC0 license ("no rights reserved").

**Notice**: if you you consider that our data contains material that is owned by you and should therefore not be reproduced here, please contact us
