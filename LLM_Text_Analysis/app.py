"""
LLM Text Analysis Application

This application demonstrates various text analysis capabilities using
modern NLP techniques and simulated LLM functionality.

Features:
- Sentiment Analysis
- Text Summarization
- Key Phrase Extraction
- Text Generation
- Language Translation (simulation)
"""

import streamlit as st
import re
from collections import Counter
from typing import List, Dict
import random

class TextAnalyzer:
    """Text analysis with simulated LLM capabilities"""
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        # Simple sentiment analysis based on word lists
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'best', 'happy', 'joy', 'perfect', 'awesome',
            'brilliant', 'positive', 'success', 'beautiful', 'incredible'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
            'poor', 'negative', 'sad', 'anger', 'fail', 'failure', 'ugly',
            'disappointing', 'disappointed', 'wrong', 'problem'
        }
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            sentiment = "Neutral"
            confidence = 0.5
        else:
            if pos_count > neg_count:
                sentiment = "Positive"
                confidence = pos_count / total
            elif neg_count > pos_count:
                sentiment = "Negative"
                confidence = neg_count / total
            else:
                sentiment = "Neutral"
                confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_words': pos_count,
            'negative_words': neg_count,
            'details': f"Found {pos_count} positive and {neg_count} negative indicators"
        }
    
    def summarize_text(self, text: str, num_sentences: int = 3) -> str:
        """Extract key sentences as summary"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Score sentences by length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Favor longer sentences and those at the beginning
            score = len(sentence.split()) * (1 + 1/(i+1))
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top N
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        summary_sentences = [s for s, _ in scored_sentences[:num_sentences]]
        
        # Return in original order
        result = []
        for sentence in sentences:
            if sentence in summary_sentences:
                result.append(sentence)
        
        return '. '.join(result) + '.'
    
    def extract_key_phrases(self, text: str, top_n: int = 10) -> List[tuple]:
        """Extract key phrases from text"""
        # Remove punctuation and convert to lowercase
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        words = cleaned.split()
        
        # Remove common stop words
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'this', 'that',
            'it', 'from', 'be', 'are', 'was', 'were', 'been', 'being'
        }
        
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Count word frequency
        word_freq = Counter(filtered_words)
        
        return word_freq.most_common(top_n)
    
    def generate_text(self, prompt: str, style: str = "creative") -> str:
        """Generate text continuation (simulated)"""
        # This is a simple simulation. In production, use actual LLM API
        
        templates = {
            "creative": [
                "continues with an innovative perspective...",
                "explores new dimensions of this concept...",
                "takes an unexpected turn, revealing...",
                "unfolds into a fascinating exploration of..."
            ],
            "professional": [
                "demonstrates significant implications for...",
                "provides valuable insights into...",
                "establishes a framework for understanding...",
                "highlights the importance of..."
            ],
            "casual": [
                "basically means that...",
                "is pretty interesting because...",
                "makes you think about...",
                "reminds us that..."
            ]
        }
        
        responses = templates.get(style, templates["creative"])
        continuation = random.choice(responses)
        
        # Extract keywords from prompt
        words = prompt.lower().split()
        keywords = [w for w in words if len(w) > 4][:3]
        
        return f"{prompt} {continuation} The key concepts of {', '.join(keywords)} are particularly relevant in modern contexts. [Note: This is a simulated response. In production, this would use an actual LLM API like OpenAI GPT-4, Anthropic Claude, or similar.]"
    
    def translate_text(self, text: str, target_lang: str) -> str:
        """Simulate translation (in production, use actual API)"""
        return f"[Translated to {target_lang}]: {text}\n\n[Note: This is a placeholder. In production, integrate Google Translate API, DeepL, or similar service.]"


def main():
    st.set_page_config(
        page_title="LLM Text Analysis",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 LLM-Powered Text Analysis")
    st.markdown("""
    ### Modern NLP & Text Analysis Tools
    
    Explore various text analysis capabilities powered by NLP and LLM techniques:
    - Sentiment Analysis
    - Text Summarization
    - Key Phrase Extraction
    - Text Generation
    - Language Translation
    
    **Note**: This demo uses simplified algorithms. For production, integrate with:
    - OpenAI GPT-4 API
    - Anthropic Claude API
    - Hugging Face Transformers
    - Google Cloud NLP
    """)
    
    # Initialize analyzer
    analyzer = TextAnalyzer()
    
    # Sidebar
    st.sidebar.header("🎯 Analysis Options")
    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        ["Sentiment Analysis", "Text Summarization", "Key Phrases", 
         "Text Generation", "Translation"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### API Integration Guide
    
    To use real LLM APIs:
    
    **OpenAI:**
    ```python
    import openai
    openai.api_key = 'your-key'
    response = openai.ChatCompletion.create(...)
    ```
    
    **Anthropic Claude:**
    ```python
    import anthropic
    client = anthropic.Client('your-key')
    response = client.messages.create(...)
    ```
    """)
    
    # Main content area
    if analysis_type == "Sentiment Analysis":
        st.header("😊 Sentiment Analysis")
        
        st.markdown("""
        Analyze the emotional tone of text. Useful for:
        - Customer feedback analysis
        - Social media monitoring
        - Product reviews
        - Brand sentiment tracking
        """)
        
        # Sample texts
        sample_texts = {
            "Positive Review": "This product is absolutely amazing! The quality exceeded my expectations and the customer service was fantastic. I'm very happy with my purchase and would definitely recommend it to others.",
            "Negative Review": "Very disappointed with this purchase. The product quality was terrible and it broke after just a few days. Customer service was unhelpful. Would not recommend.",
            "Neutral Review": "The product arrived on time and matches the description. It works as expected. Nothing particularly special but it does the job."
        }
        
        sample_choice = st.selectbox("Use sample text:", ["Custom"] + list(sample_texts.keys()))
        
        if sample_choice == "Custom":
            text_input = st.text_area("Enter text to analyze:", height=150,
                                     placeholder="Enter your text here...")
        else:
            text_input = st.text_area("Enter text to analyze:", value=sample_texts[sample_choice], height=150)
        
        if st.button("Analyze Sentiment", type="primary"):
            if text_input:
                with st.spinner("Analyzing sentiment..."):
                    result = analyzer.analyze_sentiment(text_input)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", result['sentiment'])
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    with col3:
                        sentiment_emoji = {
                            "Positive": "😊",
                            "Negative": "😞",
                            "Neutral": "😐"
                        }
                        st.metric("Indicator", sentiment_emoji[result['sentiment']])
                    
                    st.info(result['details'])
                    
                    # Visualization
                    st.markdown("---")
                    st.markdown("### Word Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Positive Words", result['positive_words'])
                    with col2:
                        st.metric("Negative Words", result['negative_words'])
            else:
                st.warning("Please enter text to analyze!")
    
    elif analysis_type == "Text Summarization":
        st.header("📋 Text Summarization")
        
        st.markdown("""
        Automatically generate concise summaries of longer texts. Perfect for:
        - Document summarization
        - Article condensation
        - Meeting notes
        - Research paper abstracts
        """)
        
        sample_text = """
        Artificial intelligence has made remarkable progress in recent years, transforming various industries
        and aspects of daily life. Machine learning algorithms can now process vast amounts of data to identify
        patterns and make predictions with unprecedented accuracy. Natural language processing has enabled 
        computers to understand and generate human language, leading to sophisticated chatbots and virtual 
        assistants. Computer vision systems can recognize objects, faces, and scenes in images and videos,
        powering applications from autonomous vehicles to medical diagnosis. Deep learning, a subset of machine
        learning using neural networks, has been particularly revolutionary, achieving human-level performance
        in many tasks. However, these advances also raise important questions about privacy, bias, and the
        future of work. As AI continues to evolve, it's crucial to develop these technologies responsibly,
        ensuring they benefit society while minimizing potential risks. The integration of AI into various
        sectors promises increased efficiency, improved decision-making, and innovative solutions to complex
        problems. From healthcare to finance, education to entertainment, AI is reshaping how we live and work.
        """
        
        use_sample = st.checkbox("Use sample text", value=True)
        
        if use_sample:
            text_input = st.text_area("Text to summarize:", value=sample_text, height=200)
        else:
            text_input = st.text_area("Text to summarize:", height=200,
                                     placeholder="Paste your text here...")
        
        num_sentences = st.slider("Number of sentences in summary:", 1, 5, 3)
        
        if st.button("Generate Summary", type="primary"):
            if text_input:
                with st.spinner("Generating summary..."):
                    summary = analyzer.summarize_text(text_input, num_sentences)
                    
                    st.markdown("### 📝 Summary")
                    st.success(summary)
                    
                    # Stats
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Words", len(text_input.split()))
                    with col2:
                        st.metric("Summary Words", len(summary.split()))
                    with col3:
                        reduction = (1 - len(summary.split()) / len(text_input.split())) * 100
                        st.metric("Reduction", f"{reduction:.0f}%")
            else:
                st.warning("Please enter text to summarize!")
    
    elif analysis_type == "Key Phrases":
        st.header("🔑 Key Phrase Extraction")
        
        st.markdown("""
        Extract important keywords and phrases. Useful for:
        - SEO optimization
        - Content tagging
        - Topic identification
        - Document indexing
        """)
        
        text_input = st.text_area("Enter text:", height=150,
                                  placeholder="Enter text to extract key phrases from...")
        
        top_n = st.slider("Number of key phrases:", 5, 20, 10)
        
        if st.button("Extract Key Phrases", type="primary"):
            if text_input:
                with st.spinner("Extracting key phrases..."):
                    phrases = analyzer.extract_key_phrases(text_input, top_n)
                    
                    st.markdown("### 🎯 Top Key Phrases")
                    
                    # Display as table
                    import pandas as pd
                    df = pd.DataFrame(phrases, columns=['Phrase', 'Frequency'])
                    st.dataframe(df, use_container_width=True)
                    
                    # Bar chart
                    st.bar_chart(df.set_index('Phrase'))
            else:
                st.warning("Please enter text!")
    
    elif analysis_type == "Text Generation":
        st.header("✨ Text Generation")
        
        st.markdown("""
        Generate text continuations using LLM techniques. Applications:
        - Content creation
        - Creative writing
        - Brainstorming
        - Email drafting
        """)
        
        prompt = st.text_area("Enter your prompt:", height=100,
                             placeholder="Start your text here...")
        
        style = st.select_slider(
            "Generation style:",
            options=["casual", "professional", "creative"]
        )
        
        if st.button("Generate Text", type="primary"):
            if prompt:
                with st.spinner("Generating text..."):
                    generated = analyzer.generate_text(prompt, style)
                    
                    st.markdown("### 📖 Generated Text")
                    st.info(generated)
                    
                    st.warning("**Production Note**: This is a simulation. For actual text generation, use OpenAI GPT-4, Anthropic Claude, or other LLM APIs.")
            else:
                st.warning("Please enter a prompt!")
    
    else:  # Translation
        st.header("🌐 Language Translation")
        
        st.markdown("""
        Translate text between languages. Real-world uses:
        - Multi-language support
        - Document translation
        - Global communication
        - Content localization
        """)
        
        text_input = st.text_area("Text to translate:", height=100)
        
        target_lang = st.selectbox(
            "Target language:",
            ["Spanish", "French", "German", "Chinese", "Japanese", "Arabic"]
        )
        
        if st.button("Translate", type="primary"):
            if text_input:
                with st.spinner("Translating..."):
                    translated = analyzer.translate_text(text_input, target_lang)
                    st.info(translated)
                    
                    st.warning("**Integration Required**: Connect to Google Translate API, DeepL, or Azure Translator for actual translation.")
            else:
                st.warning("Please enter text to translate!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 Production Deployment
    
    To deploy with real LLM capabilities:
    
    1. **Choose an LLM Provider**:
       - OpenAI (GPT-4, GPT-3.5)
       - Anthropic (Claude)
       - Google (PaLM, Gemini)
       - Open-source (Llama, Mistral)
    
    2. **Set up API credentials**
    3. **Implement rate limiting and caching**
    4. **Add error handling and logging**
    5. **Monitor costs and usage**
    """)


if __name__ == "__main__":
    main()
