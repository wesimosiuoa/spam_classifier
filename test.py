
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Download required NLTK datasets
nltk.download('stopwords')
nltk.download('punkt')

# Sample data
emails = [
    ("Congratulations! You've won a free iPhone. Click here to claim your prize.", "spam"),
    ("Reminder: Your meeting is scheduled for tomorrow at 10 AM.", "not_spam"),
    ("Claim your lottery winnings now! Don't miss out!", "spam"),
    ("Hi John, can we reschedule our appointment to next week?", "not_spam"),
    ("Limited-time offer! Buy one get one free.", "spam"),
    ("Please find the attached documents for your reference.", "not_spam"),
]

# Preprocess email text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return filtered_words

# Create a feature extractor
def extract_features(words):
    return {word: True for word in words}

# Prepare the dataset
featuresets = [(extract_features(preprocess_text(email)), label) for email, label in emails]

# Split into training and testing datasets (80% training, 20% testing)
train_size = int(len(featuresets) * 0.8)
train_set, test_set = featuresets[:train_size], featuresets[train_size:]

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate the classifier
print("Accuracy:", accuracy(classifier, test_set))

# Show the most informative features
classifier.show_most_informative_features()

# Test the classifier with new emails
def classify_email(email):
    words = preprocess_text(email)
    features = extract_features(words)
    return classifier.classify(features)

# Example emails to classify
new_emails = [
    "Get a free vacation package now!",
    "Team meeting has been postponed to 3 PM tomorrow.",
]

for email in new_emails:
    print(f"Email: {email}\nClassification: {classify_email(email)}\n")
