import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from sentiment_analyzer import SentimentAnalyzer
from text_preprocessor import TextPreprocessor
from visualizations import create_sentiment_distribution, create_sentiment_timeline, create_word_cloud
import io

# Page configuration
st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def load_analyzer():
    return SentimentAnalyzer()

@st.cache_resource
def load_preprocessor():
    return TextPreprocessor()

def main():
    st.title("📊 Social Media Sentiment Analysis Tool")
    st.markdown("Analyze sentiment in social media posts using advanced NLP techniques")
    
    analyzer = load_analyzer()
    preprocessor = load_preprocessor()
    
    # Sidebar for options
    st.sidebar.header("Analysis Options")
    analysis_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Single Text Analysis", "Batch Text Analysis", "Upload CSV File"]
    )
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    
    if analysis_mode == "Single Text Analysis":
        single_text_analysis(analyzer, preprocessor)
    elif analysis_mode == "Batch Text Analysis":
        batch_text_analysis(analyzer, preprocessor)
    else:
        file_upload_analysis(analyzer, preprocessor)
    
    # Display results if available
    if st.session_state.analysis_results:
        display_analysis_results()

def single_text_analysis(analyzer, preprocessor):
    st.header("Single Text Analysis")
    
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste your social media text here..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_button = st.button("Analyze Sentiment", type="primary")
    
    with col2:
        if st.button("Clear Results"):
            st.session_state.analysis_results = []
            st.rerun()
    
    if analyze_button and text_input.strip():
        with st.spinner("Analyzing sentiment..."):
            # Preprocess text
            preprocessed_text = preprocessor.preprocess_text(text_input)
            
            # Analyze sentiment
            sentiment_result = analyzer.analyze_sentiment(preprocessed_text)
            
            # Add timestamp
            sentiment_result['timestamp'] = datetime.now()
            sentiment_result['original_text'] = text_input
            sentiment_result['preprocessed_text'] = preprocessed_text
            
            # Store result
            st.session_state.analysis_results.append(sentiment_result)
            
            # Display immediate result
            display_single_result(sentiment_result)

def batch_text_analysis(analyzer, preprocessor):
    st.header("Batch Text Analysis")
    
    st.markdown("Enter multiple texts separated by new lines:")
    batch_input = st.text_area(
        "Batch texts:",
        height=200,
        placeholder="Enter each text on a new line..."
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        analyze_batch = st.button("Analyze All Texts", type="primary")
    
    with col2:
        if st.button("Clear All Results"):
            st.session_state.analysis_results = []
            st.rerun()
    
    if analyze_batch and batch_input.strip():
        texts = [text.strip() for text in batch_input.split('\n') if text.strip()]
        
        if texts:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, text in enumerate(texts):
                status_text.text(f"Analyzing text {i+1} of {len(texts)}")
                
                # Preprocess and analyze
                preprocessed_text = preprocessor.preprocess_text(text)
                sentiment_result = analyzer.analyze_sentiment(preprocessed_text)
                
                # Add metadata
                sentiment_result['timestamp'] = datetime.now() - timedelta(minutes=len(texts)-i-1)
                sentiment_result['original_text'] = text
                sentiment_result['preprocessed_text'] = preprocessed_text
                
                st.session_state.analysis_results.append(sentiment_result)
                
                progress_bar.progress((i + 1) / len(texts))
            
            status_text.text("Analysis complete!")
            st.success(f"Analyzed {len(texts)} texts successfully!")

def file_upload_analysis(analyzer, preprocessor):
    st.header("CSV File Upload Analysis")
    
    st.markdown("Upload a CSV file with a 'text' column containing social media posts.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV file should have a 'text' column"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("CSV file must contain a 'text' column")
                return
            
            st.write(f"Loaded {len(df)} rows from CSV")
            st.write("Preview:")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Analyze CSV Data", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, row in df.iterrows():
                    status_text.text(f"Analyzing row {i+1} of {len(df)}")
                    
                    text = str(row['text'])
                    preprocessed_text = preprocessor.preprocess_text(text)
                    sentiment_result = analyzer.analyze_sentiment(preprocessed_text)
                    
                    # Add metadata
                    timestamp = datetime.now() - timedelta(hours=len(df)-i-1)
                    if 'timestamp' in df.columns:
                        try:
                            timestamp = pd.to_datetime(row['timestamp'])
                        except:
                            pass
                    
                    sentiment_result['timestamp'] = timestamp
                    sentiment_result['original_text'] = text
                    sentiment_result['preprocessed_text'] = preprocessed_text
                    
                    st.session_state.analysis_results.append(sentiment_result)
                    
                    progress_bar.progress((i + 1) / len(df))
                
                status_text.text("Analysis complete!")
                st.success(f"Analyzed {len(df)} texts from CSV!")
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

def display_single_result(result):
    st.subheader("Analysis Result")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_color = get_sentiment_color(result['compound_score'])
        st.metric(
            "Overall Sentiment",
            result['sentiment'].title(),
            delta=f"{result['compound_score']:.3f}",
            delta_color="normal"
        )
    
    with col2:
        st.metric("Positive", f"{result['positive']:.3f}")
    
    with col3:
        st.metric("Negative", f"{result['negative']:.3f}")
    
    with col4:
        st.metric("Neutral", f"{result['neutral']:.3f}")
    
    # Show preprocessing details
    with st.expander("Preprocessing Details"):
        st.write("**Original Text:**")
        st.write(result['original_text'])
        st.write("**Preprocessed Text:**")
        st.write(result['preprocessed_text'])

def display_analysis_results():
    if not st.session_state.analysis_results:
        return
    
    st.header("📈 Analysis Results Dashboard")
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(st.session_state.analysis_results)
    
    # Summary metrics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Texts", len(df_results))
    
    with col2:
        positive_count = len(df_results[df_results['sentiment'] == 'positive'])
        st.metric("Positive", positive_count, f"{positive_count/len(df_results)*100:.1f}%")
    
    with col3:
        negative_count = len(df_results[df_results['sentiment'] == 'negative'])
        st.metric("Negative", negative_count, f"{negative_count/len(df_results)*100:.1f}%")
    
    with col4:
        neutral_count = len(df_results[df_results['sentiment'] == 'neutral'])
        st.metric("Neutral", neutral_count, f"{neutral_count/len(df_results)*100:.1f}%")
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Timeline", "Word Cloud", "Data Export"])
    
    with tab1:
        st.subheader("Sentiment Distribution")
        fig_dist = create_sentiment_distribution(df_results)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        st.subheader("Sentiment Timeline")
        if len(df_results) > 1:
            fig_timeline = create_sentiment_timeline(df_results)
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("Need at least 2 data points to show timeline")
    
    with tab3:
        st.subheader("Word Cloud")
        try:
            wordcloud_fig = create_word_cloud(df_results)
            st.pyplot(wordcloud_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating word cloud: {str(e)}")
    
    with tab4:
        st.subheader("Export Data")
        
        # Prepare export data
        export_df = df_results.copy()
        export_df['timestamp'] = export_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = export_df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="Download as JSON",
                data=json_data,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Display data table
        st.dataframe(
            export_df[['timestamp', 'sentiment', 'compound_score', 'positive', 'negative', 'neutral', 'original_text']], 
            use_container_width=True
        )

def get_sentiment_color(compound_score):
    if compound_score >= 0.05:
        return "green"
    elif compound_score <= -0.05:
        return "red"
    else:
        return "gray"

if __name__ == "__main__":
    main()
