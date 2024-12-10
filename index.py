print (f" Email spam classifir ")
import pandas as pd 
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter

analyzer = SentimentIntensityAnalyzer()

from nltk.tokenize import word_tokenize
dataset = pd.read_csv('email.csv', encoding='ISO-8859-1')


print (dataset.head())
print (f" \n Messages \n {dataset['Message']}")