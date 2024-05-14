# How to get the data and place it in the right folder

1. Download the data from [this link](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_reviews_spoiler.json.gz)

2. Do not unzip the file. Place the file in the raw folder under current directory.

3. Run the data_preprocessing.ipynb notebook to generate the processed data.

4. The processed data will be saved in the processed folder under current directory.

5. The processed data will be used in the model training and evaluation.

_Reference:_

- Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18.
- Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19.
