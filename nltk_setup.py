# nltk_setup.py
import nltk

NLTK_PACKAGES = ["punkt", "stopwords", "wordnet", "omw-1.4"]

for pkg in NLTK_PACKAGES:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)
