# nltk_setup.py
import nltk

def download_nltk_resources():
    packages = ["punkt", "stopwords", "wordnet", "omw-1.4"]
    for pkg in packages:
        try:
            # Try to find resource
            if pkg == "punkt":
                nltk.data.find("tokenizers/punkt")
            elif pkg == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif pkg == "wordnet":
                nltk.data.find("corpora/wordnet")
            elif pkg == "omw-1.4":
                nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download(pkg)

