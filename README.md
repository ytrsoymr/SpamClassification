
# SMS Spam Classifier Web App

## Project Overview

This project is a simple yet effective **SMS Spam Classifier** built using **Streamlit**. It utilizes a **Naive Bayes classifier** trained on a dataset of SMS messages to classify user-submitted messages as either "Spam" or "Not Spam." The project follows an end-to-end data science workflow, from data preprocessing to model deployment as a web application.

## Features

- **Data Preprocessing:** Clean and preprocess text data by removing special characters, converting text to lowercase, and removing stopwords.
- **Bag of Words Model:** Use `CountVectorizer` to convert text data into numerical vectors.
- **Naive Bayes Model:** Train a **Multinomial Naive Bayes** model for spam classification.
- **User Input:** Users can enter SMS messages to check if they are spam or not.
- **Data Display:** Show examples of SMS data before and after preprocessing.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- NLTK (Natural Language Toolkit)

## Installation Instructions

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Create a Virtual Environment (Optional):**

   ```bash
   python -m venv venv
   source venv/bin/activate # For Linux/Mac
   .env\Scriptsctivate # For Windows
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset:** Place the dataset file `SMSSpamCollection` in a directory `smsspamcollection/`.

5. **Run the App:**

   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the app in your browser as prompted by Streamlit.
2. Enter an SMS message in the text input box.
3. Click the **Classify Message** button to get the classification result.
4. Optionally, click **Show Data Samples** to see examples of data before and after preprocessing.

## Data Preprocessing Steps

- Remove special characters from text.
- Convert text to lowercase.
- Remove stopwords using NLTK.
- Apply stemming using the PorterStemmer.

## Model Details

- **Vectorization:** Bag of Words using `CountVectorizer` with 2500 maximum features.
- **Classifier:** Multinomial Naive Bayes

## Sample Dataset

The dataset used for training is the classic **SMSSpamCollection**, containing labeled messages as spam or ham.

## Example Output

- "Congratulations! You've won a free iPhone! Call now!" → **Spam**
- "Let's meet tomorrow at 5 PM." → **Not Spam**

## Future Improvements

- Add support for additional classifiers such as SVM or deep learning models.
- Implement performance metrics on the app interface.
- Improve UI with more interactivity.
- Add dataset upload functionality for user customization.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
