"""
AI Chatbot with Memory

This application demonstrates a conversational AI chatbot with context memory.
It showcases modern chatbot architecture with session management and conversation history.

Features:
- Conversational interface
- Context-aware responses
- Conversation memory
- Multiple conversation threads
- Export chat history
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict
import json
import random


class ConversationalAI:
    """Chatbot with memory and context awareness"""
    
    def __init__(self):
        self.conversation_history = []
        self.context_window = 5  # Number of previous messages to consider
        
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_context(self) -> List[Dict]:
        """Get recent conversation context"""
        return self.conversation_history[-self.context_window:]
    
    def generate_response(self, user_message: str) -> str:
        """Generate a contextual response"""
        # This is a simulation. In production, use an actual LLM API
        
        user_message_lower = user_message.lower()
        
        # Intent recognition based on keywords
        greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good evening']
        farewells = ['bye', 'goodbye', 'see you', 'farewell']
        questions_about_bot = ['who are you', 'what are you', 'what can you do', 'help']
        
        # Check context for follow-up questions
        context = self.get_context()
        has_context = len(context) > 1
        
        # Generate response based on intent and context
        if any(greeting in user_message_lower for greeting in greetings):
            if has_context:
                return random.choice([
                    "Hello again! How can I help you?",
                    "Hi! What would you like to talk about?",
                    "Hey! I'm here to help. What's on your mind?"
                ])
            else:
                return random.choice([
                    "Hello! I'm an AI assistant. How can I help you today?",
                    "Hi there! I'm here to help. What would you like to know?",
                    "Greetings! Ask me anything and I'll do my best to assist you."
                ])
        
        elif any(farewell in user_message_lower for farewell in farewells):
            return random.choice([
                "Goodbye! Feel free to return anytime.",
                "See you later! It was nice chatting with you.",
                "Farewell! Don't hesitate to come back if you need help."
            ])
        
        elif any(q in user_message_lower for q in questions_about_bot):
            return """I'm an AI chatbot with conversational memory. I can:
            
- Answer questions and have conversations
- Remember context from our chat
- Provide information on various topics
- Help with brainstorming and problem-solving

I'm currently running in demo mode. In production, I would be powered by advanced LLMs like GPT-4, Claude, or similar models.

What would you like to talk about?"""
        
        # Context-aware responses
        elif 'what' in user_message_lower and 'about' in user_message_lower and has_context:
            prev_topic = context[-2]['content'] if len(context) >= 2 else "our previous discussion"
            return f"Regarding {prev_topic}, I can provide more details or we can explore related topics. What specifically interests you?"
        
        elif '?' in user_message:
            # It's a question
            responses = [
                f"That's an interesting question about '{user_message}'. In a production environment, I would use an LLM API to provide a detailed answer based on my training data and the conversation context.",
                f"Great question! To properly answer '{user_message}', I would analyze it using advanced language models and provide a comprehensive response.",
                f"I understand you're asking about: {user_message}. With full LLM integration, I could give you a detailed, context-aware answer."
            ]
            return random.choice(responses)
        
        else:
            # General response
            responses = [
                f"I understand you mentioned: '{user_message}'. With full LLM capabilities, I would provide a thoughtful, context-aware response.",
                f"Thanks for sharing that. In production, I would analyze your message using advanced NLP and generate an appropriate response.",
                f"I see. Let me think about '{user_message}'. With real LLM integration, I could engage in a more natural conversation about this topic."
            ]
            
            if has_context:
                responses.append(f"Building on our conversation, regarding '{user_message}', I would provide insights that connect to what we've discussed.")
            
            return random.choice(responses) + "\n\n[Note: This is a demo. For production, integrate with OpenAI, Anthropic, or other LLM APIs.]"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


def main():
    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for chat interface
    st.markdown("""
    <style>
    .user-message {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .timestamp {
        font-size: 0.8em;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 AI Chatbot with Memory")
    st.markdown("""
    ### Conversational AI with Context Awareness
    
    This chatbot demonstrates:
    - **Context Memory**: Remembers previous messages
    - **Natural Conversations**: Responds based on conversation flow
    - **Session Management**: Maintains conversation state
    - **Export Capability**: Download chat history
    
    **Production Note**: This demo uses simulated responses. For real deployment,
    integrate with OpenAI GPT-4, Anthropic Claude, or other LLM APIs.
    """)
    
    # Initialize chatbot in session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ConversationalAI()
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("💬 Chat Settings")
        
        # Context window setting
        context_size = st.slider(
            "Context Window",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of previous messages to consider"
        )
        st.session_state.chatbot.context_window = context_size
        
        st.markdown("---")
        
        # Conversation stats
        st.markdown("### 📊 Conversation Stats")
        total_messages = len(st.session_state.messages)
        user_messages = sum(1 for m in st.session_state.messages if m['role'] == 'user')
        st.metric("Total Messages", total_messages)
        st.metric("Your Messages", user_messages)
        st.metric("Bot Messages", total_messages - user_messages)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chatbot.clear_history()
            st.session_state.messages = []
            st.rerun()
        
        # Export chat
        if st.session_state.messages:
            chat_export = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                label="📥 Download Chat",
                data=chat_export,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Sample prompts
        st.markdown("### 💡 Try These")
        sample_prompts = [
            "Hello! Who are you?",
            "What can you help me with?",
            "Tell me about AI",
            "How does machine learning work?",
            "What are your capabilities?"
        ]
        
        for prompt in sample_prompts:
            if st.button(prompt, key=f"sample_{prompt}", use_container_width=True):
                st.session_state.pending_message = prompt
                st.rerun()
    
    # Main chat area
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamp" in message:
                    st.caption(f"_{message['timestamp']}_")
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Check for pending message from sidebar
        if hasattr(st.session_state, 'pending_message'):
            prompt = st.session_state.pending_message
            delattr(st.session_state, 'pending_message')
        
        # Add user message
        timestamp = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        st.session_state.chatbot.add_message("user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"_{timestamp}_")
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.generate_response(prompt)
                st.markdown(response)
                timestamp = datetime.now().strftime("%I:%M %p")
                st.caption(f"_{timestamp}_")
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": timestamp
        })
        st.session_state.chatbot.add_message("assistant", response)
        
        st.rerun()
    
    # Check for pending message
    if hasattr(st.session_state, 'pending_message'):
        prompt = st.session_state.pending_message
        delattr(st.session_state, 'pending_message')
        
        # Add user message
        timestamp = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        st.session_state.chatbot.add_message("user", prompt)
        
        # Generate response
        response = st.session_state.chatbot.generate_response(prompt)
        timestamp = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": timestamp
        })
        st.session_state.chatbot.add_message("assistant", response)
        
        st.rerun()
    
    # Information section
    with st.expander("ℹ️ About This Chatbot"):
        st.markdown("""
        ### How It Works
        
        1. **User Input**: You type a message
        2. **Context Retrieval**: Bot retrieves recent conversation history
        3. **Response Generation**: Generates contextual response (simulated)
        4. **Memory Update**: Stores message in conversation history
        
        ### Features
        
        - **Conversation Memory**: Remembers what you've said
        - **Context Window**: Adjustable memory size
        - **Session Persistence**: Chat persists during your session
        - **Export**: Download your conversation
        
        ### Production Integration
        
        To deploy with real LLM:
        
        ```python
        # OpenAI Example
        import openai
        
        def generate_response(messages):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages
            )
            return response.choices[0].message.content
        
        # Anthropic Claude Example
        import anthropic
        
        client = anthropic.Client(api_key="your-key")
        response = client.messages.create(
            model="claude-3-opus-20240229",
            messages=messages
        )
        ```
        
        ### Advanced Features to Add
        
        - User authentication
        - Multiple conversation threads
        - Voice input/output
        - Image understanding
        - Function calling
        - RAG integration for domain knowledge
        - Custom personality/system prompts
        """)


if __name__ == "__main__":
    main()
