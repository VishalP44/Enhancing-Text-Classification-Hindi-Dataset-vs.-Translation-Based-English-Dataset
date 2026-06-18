# Hindi Text Classification vs. Translation-Based English Classification

Comparing four ML/DL models on a native Hindi cyberbullying-detection dataset against the same dataset machine-translated into English, to measure how much accuracy translation costs (or gains) for a low-resource language.

> MSc Big Data Science dissertation, Queen Mary University of London (Distinction). Full write-up in [`PAPER.pdf`](./PAPER.pdf).

## Research question

Is it better to train NLP models directly on native-language (Hindi) text, or to machine-translate the data into English first and use the more mature English-language NLP tooling? This is a practical question for any team building NLP systems for a non-English market: translate-then-classify, or build native pipelines?

## Approach

- **Dataset**: the "Hostility Detection Dataset in Hindi" (Bhardwaj, Akhtar, Ekbal, Das & Chakraborty) — Hindi social media posts labelled non-hostile, hate, offensive, defamation, or fake. Multi-label hostile categories were collapsed into a binary `hostile` / `non-hostile` task, and the majority class was downsampled to balance it (6,307 posts after cleaning).
- **Translation**: the cleaned Hindi posts were translated to English with the Google Translate API (`googletrans`) to build a parallel English dataset (`Translation.ipynb`).
- **Models**, trained and evaluated identically on both the Hindi and the translated-English versions:
  - Logistic Regression (TF-IDF features)
  - Random Forest (Word2Vec document embeddings)
  - Multinomial Naive Bayes (TF-IDF with bigrams)
  - BERT (`bert-base-multilingual-cased`, fine-tuned for 5 epochs)
- **Tech stack**: Python, scikit-learn, NLTK, Gensim (Word2Vec), PyTorch + HuggingFace Transformers (BERT), pandas/NumPy, matplotlib/seaborn. Built and run on Google Colab.

## Results

Test-set accuracy, same train/test split methodology on each language:

| Model               | Hindi | Translated English | Change |
|---------------------|:-----:|:------------------:|:------:|
| Logistic Regression  | 73.9% | 80.6%               | +6.7 pts |
| Random Forest        | 72.9% | 71.8%               | −1.1 pts |
| Naive Bayes          | 76.1% | 78.8%               | +2.7 pts |
| BERT (multilingual)  | 86.5% (ROC-AUC 0.86) | 94.7% (ROC-AUC 0.95) | +8.2 pts |

**Finding**: for this dataset, translating Hindi to English before classification *improved* accuracy for 3 of the 4 models, most notably BERT — the opposite of the "translation degrades quality" hypothesis the project set out to test. The likely explanation (discussed in the paper) is that English-language tooling, stopword lists, and especially `bert-base-multilingual-cased`'s pretraining corpus are far more weighted toward English than Hindi, so the translated pipeline benefits from better-tuned downstream components even though translation itself introduces some semantic noise. Random Forest, built on Word2Vec embeddings trained from scratch on the (small) in-domain corpus rather than a pretrained model, was the one case where the native-language version held up better.

All four models showed signs of overfitting on the Hindi data (large train/test accuracy gaps), suggesting the native-language pipelines would benefit from regularization and more data before any translation-vs-native conclusion is treated as final.

## Repository contents

- `Hindi_Dataset_Testing.ipynb` — preprocessing, label balancing, and all four models trained/evaluated on the native Hindi dataset.
- `English_Dataset_Testing.ipynb` — the same preprocessing and modeling pipeline applied to the translated English dataset.
- `Translation.ipynb` — translates the cleaned Hindi dataset to English via the Google Translate API.
- `PAPER.pdf` — the full dissertation write-up (literature review, methodology, results, discussion).

## Running it

These notebooks were built for Google Colab and read/write data from Google Drive (`/gdrive/MyDrive/...`). To run them:

1. Open a notebook in Colab, mount your own Drive, and update the hardcoded `/gdrive/MyDrive/...` paths to point at your copy of the dataset.
2. Run `Hindi_Dataset_Testing.ipynb` first — it cleans the raw dataset and produces the balanced/filtered data used downstream.
3. Run `Translation.ipynb` to generate the translated English dataset from the cleaned Hindi data.
4. Run `English_Dataset_Testing.ipynb` on the translated dataset to reproduce the English-side results.

Local execution is possible by replacing the Colab/Drive cells with local file paths and installing: `scikit-learn`, `pandas`, `numpy`, `nltk`, `gensim`, `tensorflow` (for `pad_sequences`), `torch`, `transformers`, `googletrans==4.0.0-rc1`, `matplotlib`, `seaborn`.

## Note on dataset availability

The raw and intermediate CSVs (`Dataset.csv`, `Cleaned_Dataset.csv`, `English_Translated_Dataset.csv`, `hindi_stopwords.csv`) are not included in this repository. See `PAPER.pdf` for the full dataset citation to source it independently and reproduce results end-to-end.
