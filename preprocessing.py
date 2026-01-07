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

STOPWORDS = set(stopwords.words('english'))
STOPWORDS -= {'not', 'never', 'no'}

lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> list[str]:
  # Preprocess lyrics and query, returns a tokens list

  if not isinstance(text, str):
    return []

  text = text.lower()
  text = re.sub(r"[^a-z\s]", " ", text)
  tokens = nltk.word_tokenize(text)

  tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in STOPWORDS and len(t) > 2
    ]

  return tokens