### Training Data

The training data was composed of the corpora defined in the paper. Therefore, we have two sets of training: the first one using only ParaCrawl training and the second one using domain-specific training for the WMT competition.

For the ParaCrawl, we used the ParaCrawl v6.0 (the newest one available at that time), and the training indexes used are available in the train.txt file, in this repo.

For the compora of the domain-specific corpus, we used the datasets defined in Table 3 (we will not make them available here, because of specific licenses of some datasets).

### Testing Data

We have two testing sets for this: ParaCrawl 99k and WMT
Biomedical Translation Task.

The ParaCrawl 99k is composed of 128k randomly selected sentence pairs extracted from the ParaCrawl dataset (not in the training set). Additionally, we removed similar sentences using LSH with a Jaccard similarity > 0.7. Therefore, we end up with 99k sentences, and they are available in the ParaCrawl99k folder of this repo. There is a Readme file there with further explanations. The Test_Parallel_Corpora.pkl file available in this folder has the output for Google Translate for all 128k sentences previously used. It should not be used for evaluation. Instead, use the ParaCrawl99k, since it removed duplicated/similar data.

The WMT Biomedical Translation Task pairs with ground truth are available on the WMT website, and we are not sharing the pairs here. If you wish to download it and it is not available or you are facing difficulties in extracting the pairs from it, please contact the organizers of the WMT.

