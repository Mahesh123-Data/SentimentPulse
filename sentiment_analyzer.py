from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with VADER."""
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of the given text using VADER.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Dictionary containing sentiment scores and classification
        """
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'compound_score': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'confidence': 0.0
            }
        
        # Get VADER scores
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine overall sentiment based on compound score
        compound = vader_scores['compound']
        
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Calculate confidence based on the absolute value of compound score
        confidence = abs(compound)
        
        return {
            'sentiment': sentiment,
            'compound_score': compound,
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'confidence': confidence
        }
    
    def analyze_sentiment_textblob(self, text):
        """
        Alternative sentiment analysis using TextBlob.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Dictionary containing TextBlob sentiment scores
        """
        if not text or not text.strip():
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sentiment': 'neutral'
            }
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment based on polarity
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        }
    
    def batch_analyze(self, texts):
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: List of sentiment analysis results
        """
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results
    
    def get_emotion_insights(self, text):
        """
        Get additional emotional insights from the text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Dictionary containing emotional insights
        """
        vader_scores = self.vader_analyzer.polarity_scores(text)
        textblob_scores = self.analyze_sentiment_textblob(text)
        
        # Combine insights
        insights = {
            'vader_compound': vader_scores['compound'],
            'textblob_polarity': textblob_scores['polarity'],
            'textblob_subjectivity': textblob_scores['subjectivity'],
            'emotional_intensity': abs(vader_scores['compound']),
            'objectivity': 1 - textblob_scores['subjectivity']
        }
        
        # Add interpretation
        if textblob_scores['subjectivity'] > 0.6:
            insights['text_type'] = 'highly subjective'
        elif textblob_scores['subjectivity'] > 0.3:
            insights['text_type'] = 'moderately subjective'
        else:
            insights['text_type'] = 'objective'
        
        return insights
