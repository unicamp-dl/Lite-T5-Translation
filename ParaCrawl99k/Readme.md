## ParaCrawl99k

ParaCrawl99k is a dataset that allows you to comparing your result to Google Translate (GT). It is fully available to allow you to compare your translations with 
Google Translate. Please note that we used Pt-pt translation since ParaCrawl contains mostly pt-pt translations. There are files containing 128k sentences and 99k
The 128k goes to 99k because we removed data from testing that had a jaccard similarity with any sentence in training larger than 0.7. Therefore, you can use the rest
of the 20M-120k data to train your dataset. we are going to share the final 99k as a test set. If you want the 128k version, let us know.

Google Translate metrics are available in the paper. You can also calculate yours, since there are two pickle files. Each pickle file contains the Original -> Google Translate version. It is basically a single parallel corpus of two languages but with four results (2 GT and 2 originals, each one in a mixed pair). For instance, ParaCrawl99K_EnPt_PCrawlGoogleT.pkl has English ParaCrawl version with Portuguese Google Translation and ParaCrawl99K_PtEn_PCrawlGoogleT.pkl has the ParCrawl Portuguese text in  with English Google Translate translation.

Therefore, if you want to test your own algorithm, just evaluate the first element of each tuple in the list of the pickles so you can generate your own translation. Evaluation was made using SacreBleu, as stated in the paper.

Example of tuple in ParaCrawl99K_PtEn_PCrawlGoogleT.pkl:

```
('E estou certo de que esse anseio pela democracia possa favorecer também os cristãos”.',
 'And I am sure that this yearning for democracy can also benefit Christians ”.')
```

## License

We follow ParaCrawl license scheme. Therefore, these data are released under this licensing scheme:

- We do not own any of the text from which these data has been extracted.
- We license the actual packaging of these parallel data under the Creative Commons CC0 license ("no rights reserved").

**Notice**: if you you consider that our data contains material that is owned by you and should therefore not be reproduced here, please contact us
