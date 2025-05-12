import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# 1. Load your CSV of tweets (columns: text,sentiment)
df = pd.read_csv(
     '../data/tweets.csv',
     encoding='latin-1',
     header=None,
    names=['sentiment','id','date','query','user','text'],
    on_bad_lines='skip'
)
df = df[['text','sentiment']]
# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42)

# 3. Vectorize
vect = TfidfVectorizer(stop_words='english', max_features=5000)
X_tr = vect.fit_transform(X_train)
X_te = vect.transform(X_test)

# 4. Train
clf = LogisticRegression(max_iter=1000)
clf.fit(X_tr, y_train)
print("Test accuracy:", clf.score(X_te, y_test))

# 5. Save model + vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump({'vect': vect, 'model': clf}, f)
