import nltk
from nltk.corpus import stopwords, movie_reviews
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('movie_reviews')


def preprocess_text(words):
    stop_words = set(stopwords.words('english'))
    return [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]


def extract_features(words):
    return {word: True for word in words}


documents = [(list(movie_reviews.words(fileid)), 'not_spam' if category == 'pos' else 'spam')
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

featuresets = [(extract_features(preprocess_text(words)), label) for words, label in documents]

train_size = int(len(featuresets) * 0.8)
train_set, test_set = featuresets[:train_size], featuresets[train_size:]

classifier = NaiveBayesClassifier.train(train_set)

print("Accuracy:", accuracy(classifier, test_set))

classifier.show_most_informative_features()

def classify_text(text):
    words = preprocess_text(word_tokenize(text))
    features = extract_features(words)
    return classifier.classify(features)

new_texts = [
    "Win a free iPhone by clicking here!",
    "Your meeting has been rescheduled for 10 AM tomorrow.",
    "Limited-time offer! Buy one get one free.",
    "Hi, please find the attached documents for your reference.",
]

for text in new_texts:
    print(f"Text: {text}\nClassification: {classify_text(text)}\n")
