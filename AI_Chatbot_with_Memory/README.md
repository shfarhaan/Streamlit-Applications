### AI Chatbot with Conversational Memory

A modern conversational AI chatbot demonstrating context-aware interactions and memory management.

#### Overview
This application showcases a chatbot architecture with conversation memory, enabling natural, context-aware conversations. It demonstrates key concepts in building production-ready chatbot systems.

#### Key Features

##### Core Capabilities
- 💬 **Natural Conversations**: Context-aware responses
- 🧠 **Conversation Memory**: Remembers chat history
- ⚙️ **Adjustable Context**: Configure memory window size
- 📊 **Chat Analytics**: Track conversation statistics
- 💾 **Export Chat**: Download conversation history
- 🔄 **Session Persistence**: Maintains state during session

##### User Interface
- **Modern Chat UI**: Native Streamlit chat interface
- **Timestamp Display**: Track message timing
- **Sidebar Controls**: Quick access to settings and stats
- **Sample Prompts**: Pre-written examples to get started
- **Clear Chat**: Reset conversation anytime

#### Architecture

```
User Input → Context Retrieval → Response Generation → Memory Update
                ↓                         ↓
         Previous Messages          Add to History
```

#### How It Works

1. **User sends message**
2. **System retrieves context** (last N messages based on context window)
3. **Generates response** considering conversation history
4. **Updates memory** with new user and assistant messages
5. **Displays response** with timestamp

#### Requirements
```bash
streamlit>=1.39.0
```

#### Installation & Running

```bash
# Navigate to directory
cd AI_Chatbot_with_Memory

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

#### Quick Start

1. **Start chatting**: Type in the input box at the bottom
2. **Try samples**: Click sidebar buttons for example prompts
3. **Adjust context**: Use slider to change memory window
4. **View stats**: Check conversation metrics in sidebar
5. **Export**: Download chat history as JSON

#### Demo vs Production

| Feature | Demo | Production |
|---------|------|------------|
| Response Generation | Rule-based simulation | GPT-4, Claude, Llama |
| Memory | In-memory list | Database-backed |
| Context | Simple window | Advanced RAG |
| Persistence | Session only | User accounts |
| Scalability | Single user | Multi-user |

#### Production Integration

##### OpenAI ChatGPT Integration

```python
import openai

def generate_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content

# Usage
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    *conversation_history  # Previous messages
]
response = generate_response(messages)
```

##### Anthropic Claude Integration

```python
import anthropic

client = anthropic.Client(api_key="your-key")

def generate_response(messages):
    # Convert to Claude format
    conversation = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in messages
    ])
    
    response = client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{
            "role": "user",
            "content": conversation
        }],
        max_tokens=500
    )
    return response.content[0].text
```

##### Local LLM (Llama, Mistral)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def generate_response(messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### Advanced Features

##### 1. User Authentication

```python
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(...)
name, authentication_status, username = authenticator.login()

if authentication_status:
    # Load user's conversation history
    conversations = db.get_user_conversations(username)
```

##### 2. Multiple Conversation Threads

```python
# In sidebar
conversation_list = st.sidebar.selectbox(
    "Select Conversation",
    ["New Chat"] + list(user_conversations.keys())
)

if conversation_list == "New Chat":
    st.session_state.messages = []
else:
    st.session_state.messages = user_conversations[conversation_list]
```

##### 3. RAG Integration

```python
def generate_with_rag(query, conversation_history):
    # Retrieve relevant documents
    relevant_docs = vector_db.similarity_search(query, k=3)
    
    # Augment context
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Generate response with context
    messages = conversation_history + [{
        "role": "system",
        "content": f"Use this context: {context}"
    }]
    
    return llm.generate(messages)
```

##### 4. Function Calling

```python
functions = [
    {
        "name": "search_web",
        "description": "Search the internet",
        "parameters": {...}
    },
    {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {...}
    }
]

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    functions=functions,
    function_call="auto"
)

if response.get("function_call"):
    # Execute function and return results
    function_response = execute_function(response["function_call"])
```

##### 5. Streaming Responses

```python
def stream_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        stream=True
    )
    
    placeholder = st.empty()
    full_response = ""
    
    for chunk in response:
        if chunk.choices[0].delta.get("content"):
            full_response += chunk.choices[0].delta.content
            placeholder.markdown(full_response + "▌")
    
    placeholder.markdown(full_response)
    return full_response
```

#### Best Practices

##### 1. Context Management
- **Limit context window**: Balance memory vs. token costs
- **Summarize old messages**: Compress long conversations
- **Prioritize recent**: Give more weight to recent messages

##### 2. Error Handling
```python
import time

def generate_with_retry(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm.generate(messages)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                st.error("Failed to generate response")
                return None
```

##### 3. Cost Optimization
- Use cheaper models (GPT-3.5) for simple queries
- Implement caching for common questions
- Set max token limits
- Monitor usage per user

##### 4. Safety & Moderation
```python
# Content filtering
response = openai.Moderation.create(input=user_message)
if response["results"][0]["flagged"]:
    return "I cannot respond to that request."

# Rate limiting
if user_message_count > LIMIT:
    st.warning("Rate limit exceeded. Please wait.")
```

#### Use Cases

1. **Customer Support**: 24/7 automated assistance
2. **Personal Assistant**: Scheduling, reminders, tasks
3. **Education**: Interactive tutoring
4. **Healthcare**: Symptom checker (with disclaimers)
5. **Sales**: Product recommendations
6. **HR**: Employee onboarding, FAQ
7. **Research**: Literature review assistant

#### Conversation Patterns

##### Information Retrieval
```
User: What's the weather?
Bot: [Calls weather API] Currently 72°F and sunny.
User: What about tomorrow?
Bot: [Uses context] Tomorrow will be 68°F with clouds.
```

##### Task Completion
```
User: Schedule a meeting
Bot: What time works for you?
User: 2 PM tomorrow
Bot: [Creates calendar event] Meeting scheduled for 2 PM.
```

##### Multi-turn Reasoning
```
User: I need help planning a trip
Bot: Where would you like to go?
User: Japan
Bot: When are you planning to travel?
User: Next spring
Bot: [Provides recommendations based on all context]
```

#### Performance Metrics

Track these metrics for optimization:
- Response time
- Token usage
- User satisfaction (thumbs up/down)
- Conversation length
- Drop-off rate
- Error rate

```python
# Logging example
def log_interaction(user_id, query, response, tokens, time):
    db.insert({
        'user_id': user_id,
        'query': query,
        'response': response,
        'tokens': tokens,
        'response_time': time,
        'timestamp': datetime.now()
    })
```

#### Deployment Considerations

1. **Infrastructure**
   - Use async processing for concurrent users
   - Implement load balancing
   - Set up caching layer (Redis)

2. **Security**
   - Encrypt conversation data
   - Implement authentication
   - Add rate limiting
   - Filter sensitive information

3. **Monitoring**
   - Set up logging (CloudWatch, DataDog)
   - Track errors and latency
   - Monitor API costs
   - User feedback collection

4. **Compliance**
   - GDPR: Right to deletion
   - Data retention policies
   - Conversation privacy
   - Terms of service

#### Troubleshooting

**Q: Chatbot doesn't remember context?**
- Check context window setting
- Verify messages are being stored
- Ensure session state is working

**Q: Responses are slow?**
- Use streaming responses
- Implement caching
- Choose faster model

**Q: Running out of tokens?**
- Summarize old messages
- Reduce context window
- Implement conversation pruning

#### Resources

- [OpenAI Chat Guide](https://platform.openai.com/docs/guides/chat)
- [LangChain Chat Models](https://python.langchain.com/docs/modules/model_io/chat/)
- [Streamlit Chat Elements](https://docs.streamlit.io/library/api-reference/chat)
- [Conversational AI Best Practices](https://www.rasa.com/docs/)

#### Future Enhancements

- [ ] Voice input/output
- [ ] Image understanding
- [ ] Multi-language support
- [ ] Personality customization
- [ ] Plugin system
- [ ] Analytics dashboard
- [ ] A/B testing framework

#### Credits

**Created by**: Sazzad Hussain Farhaan  
**Email**: shfarhaan21@gmail.com  
**Last Updated**: January 2026

---
*Modern conversational AI with memory using Streamlit v1.39.0+*
