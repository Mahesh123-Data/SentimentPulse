import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from collections import Counter

def create_sentiment_distribution(df):
    """
    Create a pie chart showing sentiment distribution.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
        
    Returns:
        plotly.graph_objects.Figure: Pie chart figure
    """
    sentiment_counts = df['sentiment'].value_counts()
    
    colors = {
        'positive': '#2ecc71',
        'negative': '#e74c3c', 
        'neutral': '#95a5a6'
    }
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map=colors
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        font=dict(size=12),
        height=400
    )
    
    return fig

def create_sentiment_timeline(df):
    """
    Create a timeline showing sentiment trends over time.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
        
    Returns:
        plotly.graph_objects.Figure: Timeline figure
    """
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp')
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Sentiment Over Time', 'Sentiment Scores Over Time'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Color mapping for sentiments
    color_map = {
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6'
    }
    
    # Plot 1: Sentiment categories over time
    for sentiment in df_sorted['sentiment'].unique():
        sentiment_data = df_sorted[df_sorted['sentiment'] == sentiment]
        
        fig.add_trace(
            go.Scatter(
                x=sentiment_data['timestamp'],
                y=[sentiment] * len(sentiment_data),
                mode='markers',
                name=sentiment.title(),
                marker=dict(
                    color=color_map[sentiment],
                    size=10,
                    opacity=0.7
                ),
                hovertemplate=f'<b>{sentiment.title()}</b><br>Time: %{{x}}<br>Score: %{{customdata:.3f}}<extra></extra>',
                customdata=sentiment_data['compound_score']
            ),
            row=1, col=1
        )
    
    # Plot 2: Compound scores over time
    fig.add_trace(
        go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['compound_score'],
            mode='lines+markers',
            name='Compound Score',
            line=dict(color='#3498db', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Compound Score</b><br>Time: %{x}<br>Score: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add reference lines for sentiment thresholds
    fig.add_hline(y=0.05, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_hline(y=-0.05, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title="Sentiment Analysis Timeline",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment", row=1, col=1)
    fig.update_yaxes(title_text="Compound Score", row=2, col=1)
    
    return fig

def create_word_cloud(df):
    """
    Create a word cloud from the preprocessed text.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
        
    Returns:
        matplotlib.figure.Figure: Word cloud figure
    """
    # Combine all preprocessed texts
    all_text = ' '.join(df['preprocessed_text'].dropna().astype(str))
    
    if not all_text.strip():
        # Create empty figure if no text
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No text available for word cloud', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16)
        ax.axis('off')
        return fig
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        random_state=42
    ).generate(all_text)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud of Analyzed Texts', fontsize=16, pad=20)
    
    return fig

def create_sentiment_comparison(df):
    """
    Create a comparison chart of sentiment scores.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
        
    Returns:
        plotly.graph_objects.Figure: Comparison chart
    """
    # Calculate average scores by sentiment
    avg_scores = df.groupby('sentiment')[['positive', 'negative', 'neutral']].mean()
    
    fig = go.Figure()
    
    # Add bars for each score type
    sentiments = avg_scores.index
    score_types = ['positive', 'negative', 'neutral']
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    
    for i, score_type in enumerate(score_types):
        fig.add_trace(go.Bar(
            name=score_type.title(),
            x=sentiments,
            y=avg_scores[score_type],
            marker_color=colors[i],
            hovertemplate=f'<b>%{{x}} - {score_type.title()}</b><br>Average Score: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Average Sentiment Scores by Category",
        xaxis_title="Sentiment Category",
        yaxis_title="Average Score",
        barmode='group',
        height=400
    )
    
    return fig

def create_confidence_distribution(df):
    """
    Create a histogram of sentiment confidence scores.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
        
    Returns:
        plotly.graph_objects.Figure: Histogram figure
    """
    fig = px.histogram(
        df,
        x='confidence',
        nbins=20,
        title="Distribution of Sentiment Confidence Scores",
        labels={'confidence': 'Confidence Score', 'count': 'Frequency'},
        color_discrete_sequence=['#3498db']
    )
    
    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig

def create_text_length_analysis(df):
    """
    Create a scatter plot showing relationship between text length and sentiment.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot figure
    """
    # Calculate text lengths
    df_copy = df.copy()
    df_copy['text_length'] = df_copy['original_text'].str.len()
    
    color_map = {
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6'
    }
    
    fig = px.scatter(
        df_copy,
        x='text_length',
        y='compound_score',
        color='sentiment',
        color_discrete_map=color_map,
        title="Text Length vs Sentiment Score",
        labels={'text_length': 'Text Length (characters)', 'compound_score': 'Compound Score'},
        hover_data=['confidence']
    )
    
    fig.update_layout(
        height=400,
        hovermode='closest'
    )
    
    return fig

def get_top_words_by_sentiment(df, sentiment_type, top_n=10):
    """
    Get top words for a specific sentiment type.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
        sentiment_type (str): Sentiment type ('positive', 'negative', 'neutral')
        top_n (int): Number of top words to return
        
    Returns:
        list: List of tuples (word, count)
    """
    # Filter by sentiment
    sentiment_texts = df[df['sentiment'] == sentiment_type]['preprocessed_text']
    
    # Combine all texts and split into words
    all_words = []
    for text in sentiment_texts:
        if pd.notna(text):
            words = str(text).split()
            all_words.extend([word.lower() for word in words if len(word) > 2])
    
    # Count words and return top N
    word_counts = Counter(all_words)
    return word_counts.most_common(top_n)
