### LLM Text Analysis Application

Modern text analysis powered by Natural Language Processing and LLM techniques.

#### Overview
This application demonstrates various text analysis capabilities that can be enhanced with Large Language Models (LLMs). It provides practical examples of common NLP tasks in a user-friendly interface.

#### Features

##### 5 Analysis Modes

1. **😊 Sentiment Analysis**
   - Detect emotional tone (positive, negative, neutral)
   - Confidence scoring
   - Word-level analysis
   - Use cases: Reviews, feedback, social media monitoring

2. **📋 Text Summarization**
   - Automatic text condensation
   - Adjustable summary length
   - Compression statistics
   - Use cases: Document summarization, news digests

3. **🔑 Key Phrase Extraction**
   - Identify important keywords
   - Frequency analysis
   - Visual word clouds (via frequency bars)
   - Use cases: SEO, content tagging, topic modeling

4. **✨ Text Generation**
   - Content completion
   - Multiple style options (casual, professional, creative)
   - Prompt-based generation
   - Use cases: Content creation, brainstorming, writing assistance

5. **🌐 Language Translation**
   - Multi-language support framework
   - Translation API integration guide
   - Use cases: Localization, global communication

#### Architecture

```
User Input → NLP Processing → Analysis Engine → Results Display
                ↓
         (Production: LLM API)
```

#### Requirements
```bash
streamlit>=1.39.0
pandas>=2.2.0
```

#### Installation & Running

```bash
# Navigate to directory
cd LLM_Text_Analysis

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

#### Quick Start Guide

1. **Select Analysis Type** from sidebar
2. **Enter or select sample text**
3. **Click analysis button**
4. **View results** with visualizations

#### Demo vs Production

| Component | Demo | Production |
|-----------|------|------------|
| Sentiment | Rule-based | Transformer models (BERT, RoBERTa) |
| Summary | Extractive (basic) | Abstractive (GPT-4, Claude) |
| Key Phrases | Frequency-based | NER + LLM extraction |
| Generation | Template-based | GPT-4, Claude, Llama |
| Translation | Placeholder | Google Translate, DeepL API |

#### Production Integration Examples

##### OpenAI Integration

```python
import openai

# Sentiment Analysis
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": f"Analyze sentiment: {text}"
    }]
)

# Text Summarization
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": f"Summarize in 3 sentences: {text}"
    }]
)

# Text Generation
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": f"Continue this text: {prompt}"
    }]
)
```

##### Anthropic Claude Integration

```python
import anthropic

client = anthropic.Client(api_key="your-key")

# Analysis
response = client.messages.create(
    model="claude-3-opus-20240229",
    messages=[{
        "role": "user",
        "content": f"Analyze the sentiment: {text}"
    }]
)
```

##### Hugging Face Transformers

```python
from transformers import pipeline

# Sentiment Analysis
sentiment = pipeline("sentiment-analysis")
result = sentiment(text)

# Summarization
summarizer = pipeline("summarization")
summary = summarizer(text, max_length=130, min_length=30)

# Translation
translator = pipeline("translation_en_to_fr")
translation = translator(text)
```

#### Use Cases by Feature

##### Sentiment Analysis
- **E-commerce**: Product review analysis
- **Social Media**: Brand sentiment tracking
- **Customer Service**: Feedback classification
- **Market Research**: Opinion mining

##### Text Summarization
- **News**: Article digests
- **Research**: Paper abstracts
- **Business**: Report summaries
- **Legal**: Contract highlights

##### Key Phrase Extraction
- **SEO**: Content optimization
- **Search**: Document indexing
- **Analytics**: Trend identification
- **Tagging**: Automatic categorization

##### Text Generation
- **Marketing**: Content creation
- **Email**: Response drafting
- **Creative**: Story writing
- **Code**: Documentation generation

##### Translation
- **Global Business**: Multi-language support
- **Customer Service**: International support
- **Content**: Website localization
- **Communication**: Cross-border collaboration

#### Advanced Features to Add

1. **Batch Processing**: Analyze multiple documents
2. **API Endpoints**: REST API for integration
3. **Custom Models**: Fine-tuned for specific domains
4. **A/B Testing**: Compare different prompts/models
5. **Analytics Dashboard**: Track usage and performance
6. **Export Results**: CSV, JSON, PDF reports

#### Performance Optimization

```python
# Caching for repeated analysis
@st.cache_data
def analyze_text(text, model):
    return model.analyze(text)

# Async processing for multiple requests
import asyncio

async def process_batch(texts):
    tasks = [analyze_async(text) for text in texts]
    return await asyncio.gather(*tasks)

# Rate limiting
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=10, period=60)
def call_api(text):
    return api.analyze(text)
```

#### Cost Management

When using paid APIs:

1. **Implement caching** to avoid duplicate requests
2. **Set usage limits** per user/session
3. **Use streaming** for long-form generation
4. **Choose appropriate models** (GPT-3.5 vs GPT-4)
5. **Monitor costs** with dashboards

```python
# Example cost tracking
def track_usage(user_id, tokens, cost):
    db.increment_usage(user_id, tokens, cost)
    if db.get_usage(user_id) > LIMIT:
        raise RateLimitExceeded()
```

#### Deployment Checklist

- [ ] Set up API keys securely (environment variables)
- [ ] Implement error handling and retries
- [ ] Add rate limiting
- [ ] Set up logging and monitoring
- [ ] Configure caching
- [ ] Add user authentication
- [ ] Implement usage tracking
- [ ] Set up backup/fallback models
- [ ] Create API documentation
- [ ] Deploy with proper CORS settings

#### Example Workflows

##### Content Analysis Pipeline
```
1. Upload document
2. Extract key phrases
3. Analyze sentiment
4. Generate summary
5. Export report
```

##### Multi-language Processing
```
1. Detect language
2. Translate to English
3. Perform analysis
4. Translate results back
5. Present findings
```

#### API Provider Comparison

| Provider | Best For | Pricing | Speed |
|----------|----------|---------|-------|
| OpenAI GPT-4 | Highest quality | High | Medium |
| OpenAI GPT-3.5 | Cost-effective | Low | Fast |
| Anthropic Claude | Long context | Medium | Medium |
| Google PaLM | Integration | Medium | Fast |
| Open-source | Self-hosted | Free* | Varies |

*Hardware costs apply

#### Troubleshooting

**Q: API timeout errors?**
- Implement retry logic with exponential backoff

**Q: Inconsistent results?**
- Use temperature=0 for deterministic outputs

**Q: Too expensive?**
- Cache results, use cheaper models for simple tasks

**Q: Slow performance?**
- Use async processing, batch requests

#### Resources

- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Hugging Face Course](https://huggingface.co/course)
- [LangChain Documentation](https://python.langchain.com/)
- [Anthropic Claude Docs](https://docs.anthropic.com/)

#### Next Steps

1. **Integrate** with actual LLM API
2. **Add** user authentication
3. **Implement** batch processing
4. **Create** API endpoints
5. **Deploy** to production

#### Credits

**Created by**: Sazzad Hussain Farhaan  
**Email**: shfarhaan21@gmail.com  
**Last Updated**: January 2026

---
*Modern NLP and LLM text analysis with Streamlit v1.39.0+*
