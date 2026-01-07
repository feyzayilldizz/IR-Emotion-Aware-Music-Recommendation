import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#nltk.data.path.append('/Users/feyzayildiz/nltk_data')
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def get_stopwords():
    """Return the English stopwords, download if missing"""
    try:
        sw = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        sw = set(stopwords.words("english"))
    
    sw -= {'not', 'never', 'no'}
    return sw

def ensure_wordnet():
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

def ensure_punkt():
    # Standard punkt tokenizer
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        nltk.download("punkt")


STOPWORDS = get_stopwords()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> list[str]:
  # Preprocess lyrics and query, returns a tokens list
  ensure_punkt()
  ensure_wordnet() 
  tokens = nltk.word_tokenize(text)
  if not isinstance(text, str):
    return []

  text = text.lower()
  text = re.sub(r"[^a-z\s]", " ", text)
  

  tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in STOPWORDS and len(t) > 2
    ]

  return tokens