import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with NLTK components."""
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Common social media patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        
    def _download_nltk_data(self):
        """Download required NLTK data."""
        required_data = [
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet'),
            ('corpora/omw-1.4', 'omw-1.4')
        ]
        
        for path, name in required_data:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(name, quiet=True)
    
    def remove_urls(self, text):
        """Remove URLs from text."""
        return self.url_pattern.sub('', text)
    
    def remove_mentions(self, text):
        """Remove @mentions from text."""
        return self.mention_pattern.sub('', text)
    
    def remove_hashtags(self, text):
        """Remove hashtags from text (keeps the text after #)."""
        return self.hashtag_pattern.sub(lambda m: m.group(0)[1:], text)
    
    def remove_emails(self, text):
        """Remove email addresses from text."""
        return self.email_pattern.sub('', text)
    
    def remove_phone_numbers(self, text):
        """Remove phone numbers from text."""
        return self.phone_pattern.sub('', text)
    
    def remove_extra_whitespace(self, text):
        """Remove extra whitespace and normalize spaces."""
        return re.sub(r'\s+', ' ', text).strip()
    
    def remove_punctuation(self, text, keep_emoticons=True):
        """
        Remove punctuation from text.
        
        Args:
            text (str): Input text
            keep_emoticons (bool): Whether to preserve common emoticons
            
        Returns:
            str: Text with punctuation removed
        """
        if keep_emoticons:
            # Preserve common emoticons
            emoticon_pattern = re.compile(r'[:\-;][\)\(\[\]DPpOo]+|[XxD]+|<3|</3|:\*|;\*')
            emoticons = emoticon_pattern.findall(text)
            
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Add emoticons back
            for emoticon in emoticons:
                text += ' ' + emoticon
        else:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def normalize_text(self, text):
        """Normalize text by converting to lowercase and handling contractions."""
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def tokenize_text(self, text):
        """Tokenize text into words."""
        try:
            return word_tokenize(text)
        except Exception:
            # Fallback to simple split if NLTK tokenizer fails
            return text.split()
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list."""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens to their root form."""
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception:
            # Return original tokens if lemmatization fails
            return tokens
    
    def filter_tokens(self, tokens, min_length=2):
        """Filter tokens by minimum length and remove numeric-only tokens."""
        filtered_tokens = []
        for token in tokens:
            if len(token) >= min_length and not token.isdigit():
                filtered_tokens.append(token)
        return filtered_tokens
    
    def preprocess_text(self, text, 
                       remove_urls=True,
                       remove_mentions=True, 
                       remove_hashtags=False,
                       remove_emails=True,
                       remove_phones=True,
                       remove_punctuation=False,
                       normalize=True,
                       remove_stopwords=False,
                       lemmatize=False,
                       return_tokens=False):
        """
        Complete text preprocessing pipeline.
        
        Args:
            text (str): Input text to preprocess
            remove_urls (bool): Remove URLs
            remove_mentions (bool): Remove @mentions
            remove_hashtags (bool): Remove hashtags
            remove_emails (bool): Remove email addresses
            remove_phones (bool): Remove phone numbers
            remove_punctuation (bool): Remove punctuation
            normalize (bool): Normalize text (lowercase, contractions)
            remove_stopwords (bool): Remove stopwords
            lemmatize (bool): Lemmatize words
            return_tokens (bool): Return tokens instead of text
            
        Returns:
            str or list: Preprocessed text or tokens
        """
        if not text or not isinstance(text, str):
            return [] if return_tokens else ""
        
        # Step 1: Remove unwanted elements
        if remove_urls:
            text = self.remove_urls(text)
        
        if remove_mentions:
            text = self.remove_mentions(text)
        
        if remove_hashtags:
            text = self.remove_hashtags(text)
        
        if remove_emails:
            text = self.remove_emails(text)
        
        if remove_phones:
            text = self.remove_phone_numbers(text)
        
        # Step 2: Normalize text
        if normalize:
            text = self.normalize_text(text)
        
        # Step 3: Remove punctuation (if requested)
        if remove_punctuation:
            text = self.remove_punctuation(text)
        
        # Step 4: Clean whitespace
        text = self.remove_extra_whitespace(text)
        
        # If we don't need tokens, return cleaned text
        if not remove_stopwords and not lemmatize and not return_tokens:
            return text
        
        # Step 5: Tokenization
        tokens = self.tokenize_text(text)
        
        # Step 6: Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Step 7: Lemmatization
        if lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Step 8: Filter tokens
        tokens = self.filter_tokens(tokens)
        
        # Return tokens or join back to text
        if return_tokens:
            return tokens
        else:
            return ' '.join(tokens)
    
    def extract_hashtags(self, text):
        """Extract hashtags from text."""
        return self.hashtag_pattern.findall(text)
    
    def extract_mentions(self, text):
        """Extract mentions from text."""
        return self.mention_pattern.findall(text)
    
    def get_text_statistics(self, text):
        """Get basic statistics about the text."""
        original_length = len(text)
        tokens = self.tokenize_text(text)
        
        stats = {
            'original_length': original_length,
            'word_count': len(tokens),
            'unique_words': len(set(tokens)),
            'hashtag_count': len(self.extract_hashtags(text)),
            'mention_count': len(self.extract_mentions(text)),
            'url_count': len(self.url_pattern.findall(text)),
            'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0
        }
        
        return stats
