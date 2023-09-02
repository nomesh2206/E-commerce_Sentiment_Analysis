import nltk
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
import streamlit as st
import pandas as pd
import re
import contractions
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk import FreqDist

# Set configurations
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Title and header
st.title("Sentiment Analysis Dashboard")
st.header("Exploring and Visualizing Sentiment Analysis")

# File upload
st.subheader("Upload CSV File")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display basic information about the DataFrame
    st.subheader("Basic Information about the Data")
    st.write("Number of rows and columns:", data.shape)
    st.write("Columns:", data.columns)

    # Display summary statistics for numerical columns
    st.subheader("Summary Statistics for Numerical Columns")
    st.write(data.describe())

    # Check for missing values
    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # Display rating distribution
    st.subheader("Rating Distribution")
    rating_counts = data['reviews.rating'].value_counts().sort_index()
    st.write(rating_counts)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=rating_counts.index, y=rating_counts.values)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Rating Distribution')
    st.pyplot(plt)

    # Initialize the SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()


    # Define the Lemmatizer
    lemmatizer = WordNetLemmatizer()

    def cleaner(text):
        """
        Clean and preprocess a given text using various steps.

        This function applies a series of cleaning operations to the input text, including replacing contractions,
        removing hashtags and Twitter handles, eliminating URLs, converting to lowercase, and lemmatizing words.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned and preprocessed text.
        """
        new_text = re.sub(r"'s\b", " is", text)
        new_text = re.sub("#", "", new_text)
        new_text = re.sub("@[A-Za-z0-9]+", "", new_text)
        new_text = re.sub(r"http\S+", "", new_text)
        new_text = contractions.fix(new_text)
        new_text = re.sub(r"[^a-zA-Z]", " ", new_text)
        new_text = new_text.lower().strip()

        cleaned_text = ''
        for token in new_text.split():
            cleaned_text = cleaned_text + lemmatizer.lemmatize(token) + ' '

        return cleaned_text

    # Preprocessing function
    def preprocess_text(text):
        """
        Preprocess a given text for further analysis.

        This function takes the input text, applies the 'cleaner' function, tokenizes the cleaned text,
        removes punctuation and stopwords, and then reconstructs the preprocessed text.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            str: The preprocessed text ready for analysis.
        """
        if isinstance(text, str):
            # Apply your cleaner function
            cleaned_text = cleaner(text)

            # Tokenization
            tokens = word_tokenize(cleaned_text)

            # Remove punctuation
            tokens = [token for token in tokens if token not in string.punctuation]

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]

            # Reconstruct preprocessed text
            preprocessed_text = ' '.join(tokens)
            return preprocessed_text
        else:
            # If the input is not a string, return an empty string
            return ''

    # Apply preprocess_text to 'reviews.text' column
    data['cleaned_reviews'] = data['reviews.text'].apply(preprocess_text)


        # Function to classify sentiment
    def classify_sentiment(score):
        if score['compound'] > 0.6:
            return 'positive'
        elif score['compound'] < 0.1:
            return 'negative'
        else:
            return 'neutral'
    # Calculate sentiment scores using SentimentIntensityAnalyzer
    data['sentiment_scores'] = data['cleaned_reviews'].apply(sid.polarity_scores)

    # Classify sentiment based on scores
    data['sentiment_category'] = data['sentiment_scores'].apply(classify_sentiment)


    #data['compound'] = data['reviews.text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    data['compound'] = data['reviews.text'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])



    st.subheader("Sentiment Distribution")
    sentiment_distribution = data['sentiment_category'].value_counts()
    st.write(sentiment_distribution)


    import plotly.express as px


    # Create a pie chart using Plotly
    fig = px.pie(names=sentiment_distribution.index,values=sentiment_distribution.values,title='Sentiment Distribution')

    # Display the pie chart using Streamlit
    st.plotly_chart(fig)


    
    # Display top 30 common words
    st.subheader("Top 30 Most Common Words")
    all_words = ' '.join(data['cleaned_reviews']).split()
    word_freq = Counter(all_words)

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    # Display the word cloud using Matplotlib and Streamlit
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Display the original DataFrame
    st.subheader("Original DataFrame")
    st.write(data)

else:
    st.warning("Please upload a CSV file.")

# End with a footer
st.markdown("---")
st.write("Created with ❤️ group 9")