import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Ensure stopwords are available
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Function for preprocessing text
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    return ' '.join(review)

# Load and preprocess the dataset
@st.cache_data
def load_data():
    messages = pd.read_csv(r'data/SMSSpamCollection', sep='\t', names=["label", "message"], header=None)
    corpus = [preprocess_text(message) for message in messages['message']]
    cv = CountVectorizer(max_features=2500)
    X = cv.fit_transform(corpus).toarray()
    y = pd.get_dummies(messages['label']).iloc[:, 1].values
    return messages, corpus, X, y, cv

# Load data and train the model
messages, corpus, X, y, cv = load_data()

# Split and train the Naive Bayes model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

spam_detect_model = MultinomialNB().fit(X_train, y_train)

# Streamlit App
st.title("SMS Spam Classifier")

st.write("This app uses a Naive Bayes classifier to detect spam messages.")

# Display original and preprocessed data
st.subheader("Sample Data Before and After Preprocessing")
if st.button("Show Data Samples"):
    sample_data = pd.DataFrame({"Original Message": messages['message'][:5], "Preprocessed Message": corpus[:5]})
    st.dataframe(sample_data)

user_input = st.text_area("Enter an SMS message to classify:")

if st.button("Classify Message"):
    if user_input.strip():
        # Preprocess the input
        processed_input = preprocess_text(user_input)
        input_vector = cv.transform([processed_input]).toarray()
        prediction = spam_detect_model.predict(input_vector)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        st.success(f"This message is classified as: {result}")
    else:
        st.warning("Please enter a valid message.")
