# B1 Chatbot - AI-Powered Real Estate Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Django](https://img.shields.io/badge/Django-4.x-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991.svg)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An intelligent AI-powered conversational assistant built specifically for **B1 Properties**, Dubai's premier luxury real estate brokerage. This chatbot leverages **RAG (Retrieval-Augmented Generation)** architecture with OpenAI's GPT-4 and LangChain to provide accurate, context-aware responses about Dubai's luxury property market.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [RAG Architecture](#rag-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Telegram Bot Setup](#telegram-bot-setup)
- [Database Schema](#database-schema)
- [API Integration](#api-integration)
- [Deployment](#deployment)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**B1 Chatbot** is not just another chatbot - it's an intelligent assistant that combines cutting-edge AI technology with domain-specific knowledge about Dubai's luxury real estate market. Built for B1 Properties (founded by Babak Jafari), it specializes in ultra-luxury properties in prime locations like:

- Palm Jumeirah
- Jumeirah Bay
- Bluewaters Island
- Other premium Dubai locations

### What Makes This Special?

1. **RAG Architecture**: Uses Retrieval-Augmented Generation to provide accurate, company-specific information
2. **Custom Knowledge Base**: Trained on proprietary B1 Properties data
3. **Multi-Platform**: Web interface + Telegram bot integration
4. **Production-Ready**: Deployed on Vercel with full error handling

---

## Key Features

### Core Functionality

#### Intelligent Conversational AI
- **Context-Aware Responses**: Understands real estate terminology and context
- **Custom Knowledge Base**: Proprietary Dubai property market data
- **Query Classification**: Automatically categorizes questions into:
  - Transaction statistics
  - Home prices and market values
  - Market insights and predictions
  - Location/community navigation

#### RAG (Retrieval-Augmented Generation)
- **Semantic Search**: Vector-based document retrieval
- **Document Chunking**: Intelligent text splitting for optimal context
- **Embedding Generation**: OpenAI's text-embedding models
- **Vector Database**: Chroma for efficient similarity search
- **Response Enhancement**: Query-type specific information enrichment

#### User Management
- User registration and authentication
- Secure login/logout functionality
- Session management
- Password validation
- User-specific chat history

#### Chat History Management
- **Persistent Storage**: All conversations saved in database
- **User-Specific History**: Each user sees only their conversations
- **Timestamped Messages**: Track conversation flow
- **Delete History**: Privacy-focused clear all messages feature

#### Multi-Platform Support
- **Web Interface**: Responsive Django-based chat UI
- **Telegram Integration**: Mobile-friendly bot access
- **Real-Time Responses**: Asynchronous message processing

#### Error Handling & Monitoring
- OpenAI API connection error detection
- Rate limit warnings with user feedback
- Graceful fallbacks for API failures
- Django messages framework for notifications

---

## Tech Stack

### Backend Framework
- **Django 4.x** - Python web framework
- **Python 3.10+** - Core programming language
- **SQLite** - Development database (PostgreSQL/MySQL ready)

### AI & Machine Learning
- **OpenAI GPT-4** - Large Language Model for natural language understanding
- **LangChain** - Framework for building LLM applications
- **LangChain-OpenAI** - OpenAI integration for LangChain
- **LangChain-Chroma** - Vector database integration
- **LangChain-Core** - Core LangChain functionality
- **LangChain-Text-Splitters** - Document chunking utilities

### Vector Database
- **Chroma** - Efficient vector storage and similarity search
- **OpenAI Embeddings** - Text-to-vector transformation

### Integrations
- **Python-Telegram-Bot** - Telegram bot API wrapper
- **Python-Dotenv** - Environment variable management
- **Requests** - HTTP library for API calls

### Frontend
- **Django Templates** - Server-side rendering
- **Bootstrap** - Responsive UI framework
- **Vanilla JavaScript** - Interactive features

### Deployment
- **Vercel** - Serverless deployment platform
- **WSGI/ASGI** - Production server interfaces

---

## RAG Architecture

### What is RAG?

**Retrieval-Augmented Generation (RAG)** is an advanced AI architecture that combines:
1. **Information Retrieval** - Finding relevant documents
2. **Generation** - Creating natural language responses

### Why RAG Over Standard LLM?

| Standard LLM | RAG-Enhanced LLM |
|--------------|------------------|
| Limited to training data | Access to custom knowledge base |
| Can hallucinate facts | Grounded in real documents |
| Static knowledge | Dynamic, updatable information |
| Generic responses | Domain-specific accuracy |

### Our RAG Pipeline

```
User Query
    ↓
1. Query Processing & Classification
    ↓
2. Vector Embedding Generation
    ↓
3. Semantic Search in Chroma DB
    ↓
4. Retrieve Relevant Context (Top-K Documents)
    ↓
5. Construct Prompt with Context
    ↓
6. GPT-4 Response Generation
    ↓
7. Response Enhancement (Query-Type Specific)
    ↓
8. Return Final Answer
```

### Implementation Details

**Document Processing** (openapp/rag.py:33-38):
```python
# Load custom knowledge base
with open('dubai_property.txt', "r") as file:
    text = file.read().split("##chunk##")
documents = [Document(page_content=chunk) for chunk in text]
```

**Embedding & Vector Store** (openapp/rag.py:40-48):
```python
# Chunking for optimal context
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
chunks = text_splitter.split_documents(documents)

# Generate embeddings and store in Chroma
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)
```

**RAG Chain** (openapp/rag.py:50-76):
```python
retriever = vectorstore.as_retriever()

# Custom B1 Properties prompt
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PromptTemplate.from_template(b1_prompt)
    | llm
    | StrOutputParser()
)
```

---

## Project Structure

```
b1_chatbot-main/
├── openapp/                              # Main Django application
│   ├── models.py                         # Database models (ChatGptBot)
│   ├── views.py                          # View logic (HomeView, auth views)
│   ├── rag.py                            # RAG implementation (RAG_CLS)
│   ├── telgrambot.py                     # Telegram bot integration
│   ├── forms.py                          # User forms (SignUp, Login)
│   ├── urls.py                           # App URL routing
│   ├── admin.py                          # Django admin configuration
│   ├── dubai_property.txt                # Knowledge base document
│   ├── apps.py                           # App configuration
│   ├── tests.py                          # Test cases
│   └── migrations/                       # Database migrations
│       ├── 0001_initial.py
│       └── 0002_alter_chatgptbot_options_chatgptbot_created_at.py
├── coreOpenai/                           # Django project settings
│   ├── settings.py                       # Configuration & installed apps
│   ├── urls.py                           # Main URL routing
│   ├── wsgi.py                           # WSGI configuration
│   ├── asgi.py                           # ASGI configuration
│   └── __init__.py
├── templates/                            # HTML templates
│   ├── base.html                         # Base template with navbar
│   ├── index.html                        # Chat interface
│   ├── login.html                        # Login page
│   └── users/                            # User-related templates
│       └── register.html                 # Registration page
├── static/                               # Static files (CSS, JS, images)
│   ├── css/
│   ├── js/
│   └── images/
├── requirements.txt                      # Python dependencies
├── req.txt                               # Alternative requirements file
├── vercel.json                           # Vercel deployment config
├── manage.py                             # Django management script
├── .gitignore                            # Git ignore rules
├── LICENSE                               # MIT License
└── README.md                             # This file
```

---

## Installation

### Prerequisites

- **Python 3.10 or higher**
- **pip** (Python package manager)
- **Git** (for version control)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **Telegram Bot Token** (optional, for Telegram integration) ([Create bot with BotFather](https://core.telegram.org/bots#botfather))

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/b1_chatbot.git
cd b1_chatbot-main
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages installed:**
- Django
- openai>=1.40.0
- langchain
- langchain-chroma
- langchain-openai
- langchain-core
- langchain-text-splitters
- python-telegram-bot
- python-dotenv

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
TELEGRAM_TOKEN=your_telegram_bot_token_here  # Optional
SECRET_KEY=your_django_secret_key_here
DEBUG=True
```

**How to get your OpenAI API Key:**
1. Sign up at [OpenAI Platform](https://platform.openai.com/)
2. Navigate to API Keys section
3. Create a new secret key
4. Copy and paste into `.env` file

### Step 5: Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### Step 6: Create Superuser (Admin Account)

```bash
python manage.py createsuperuser
```

Follow the prompts to create an admin account.

### Step 7: Run Development Server

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` in your browser.

---

## Configuration

### Django Settings (coreOpenai/settings.py)

#### Database Configuration

**Development (Default - SQLite):**
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

**Production (PostgreSQL):**
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME'),
        'USER': os.getenv('DB_USER'),
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': os.getenv('DB_HOST'),
        'PORT': '5432',
    }
}
```

#### OpenAI Configuration

The RAG system is initialized in `openapp/views.py:21`:

```python
from .rag import RAG_CLS

# Initialize RAG model with knowledge base
b1_model = RAG_CLS('dubai_property.txt')
```

**Customize RAG settings** in `openapp/rag.py:24-32`:

```python
class RAG_CLS():
    def __init__(self, pth):
        self.llm = ChatOpenAI(model="gpt-4")  # Change model here
        self.pth = os.path.join(os.path.dirname(__file__), pth)
        self.documents = self.load_data()
        self.vectorstore = self.genrate_embedings()
        self.rag_chain = self.get_chain()
```

**Available OpenAI models:**
- `gpt-4` (Recommended - most accurate)
- `gpt-4-turbo`
- `gpt-3.5-turbo` (Faster, cheaper, less accurate)

#### Chunk Size Configuration

Adjust in `openapp/rag.py:42`:

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
```

**Chunk size considerations:**
- **Smaller chunks** (500-1000): More precise retrieval, may miss context
- **Larger chunks** (1500-2000): Better context, may retrieve irrelevant info
- **Default**: 1500 (balanced)

### Security Settings (Production)

Update `coreOpenai/settings.py`:

```python
DEBUG = False
SECRET_KEY = os.getenv('SECRET_KEY')  # Never hardcode in production
ALLOWED_HOSTS = ['yourdomain.com', '.vercel.app']
```

---

## Usage

### Web Interface

#### 1. Register/Login

- **Register**: Navigate to `/sign-up/` or click "Sign Up" button
- **Login**: Navigate to `/login/` or click "Login" button

#### 2. Chat Interface

Once logged in, you'll see:
- **Chat history** on the left/top (previous conversations)
- **Input box** at the bottom for new messages
- **Send button** to submit queries

#### 3. Example Queries

**Property Information:**
```
"Tell me about luxury villas in Palm Jumeirah"
"What properties does B1 Properties offer in Bluewaters Island?"
"Show me ultra-luxury apartments in Dubai"
```

**Market Insights:**
```
"What are the current price trends in Dubai real estate?"
"Is now a good time to invest in Dubai properties?"
"What's the market outlook for luxury properties?"
```

**Transaction Statistics:**
```
"What was B1 Properties' biggest sale?"
"Show me recent transaction data"
"What's the average price per square foot in Palm Jumeirah?"
```

**About B1 Properties:**
```
"Tell me about B1 Properties"
"Who is Babak Jafari?"
"What services does B1 Properties provide?"
```

#### 4. Manage Chat History

- **View History**: Scroll through past conversations
- **Delete History**: Click "Delete History" button to clear all messages

#### 5. Logout

Click "Logout" button in the navigation bar.

---

## Telegram Bot Setup

### Step 1: Create Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` command
3. Follow prompts to name your bot
4. Copy the **HTTP API token**

### Step 2: Configure Environment

Add token to `.env`:
```
TELEGRAM_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
```

### Step 3: Run Telegram Bot

```bash
python openapp/telgrambot.py
```

### Step 4: Interact with Bot

1. Search for your bot in Telegram
2. Send `/start` command
3. Ask questions about Dubai real estate

**Example conversation:**
```
You: /start
Bot: Hi, i'm ready to assist you, please ask anything you want!

You: Tell me about Palm Jumeirah properties
Bot: [AI-generated response about Palm Jumeirah properties]
```

---

## Database Schema

### Models (openapp/models.py)

#### ChatGptBot Model

Stores all chat conversations.

| Field | Type | Description |
|-------|------|-------------|
| id | AutoField | Primary key (auto-generated) |
| user | ForeignKey(User) | Reference to Django User model |
| messageInput | TextField | User's question/message |
| bot_response | TextField | AI-generated response |
| created_at | DateTimeField | Timestamp (auto-generated) |

**Meta Options:**
```python
class Meta:
    verbose_name = 'Messages History'
    verbose_name_plural = 'Messages History'
    ordering = ['created_at']  # Chronological order
```

**String Representation:**
```python
def __str__(self):
    return self.user.username
```

### Database Queries

**Get user's chat history:**
```python
user_chats = ChatGptBot.objects.filter(user=request.user)
```

**Delete all user's messages:**
```python
ChatGptBot.objects.filter(user=request.user).delete()
```

**Create new chat entry:**
```python
ChatGptBot.objects.create(
    user=request.user,
    messageInput="User's question",
    bot_response="AI response"
)
```

---

## API Integration

### OpenAI API

#### Configuration (openapp/rag.py)

```python
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

#### Models Used

1. **GPT-4** (Text Generation):
   ```python
   llm = ChatOpenAI(model="gpt-4")
   ```

2. **Text-Embedding-Ada-002** (Embeddings):
   ```python
   embeddings = OpenAIEmbeddings()
   ```

#### Error Handling (openapp/views.py:54-59)

```python
try:
    response = b1_model.get_answer(clean_user_input)
except openai.APIConnectionError as e:
    messages.warning(request, "Failed to connect to OpenAI API")
except openai.RateLimitError as e:
    messages.warning(request, "You exceeded your current quota")
```

### API Cost Optimization

**Estimated costs per 1,000 queries:**
- GPT-4: $0.03 (input) + $0.06 (output)
- Embeddings: $0.0001

**Cost-saving tips:**
1. Use `gpt-3.5-turbo` for less critical queries
2. Implement response caching for common questions
3. Limit chat history context length
4. Set max_tokens for responses

---

## Deployment

### Vercel Deployment

The project includes `vercel.json` configuration for easy deployment.

#### Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

#### Step 2: Login to Vercel

```bash
vercel login
```

#### Step 3: Deploy

```bash
vercel --prod
```

#### Step 4: Set Environment Variables

In Vercel dashboard:
1. Go to Project Settings > Environment Variables
2. Add:
   - `OPENAI_API_KEY`
   - `SECRET_KEY`
   - `TELEGRAM_TOKEN` (if using Telegram)

### Alternative Deployment Options

#### Heroku

```bash
# Install Heroku CLI
heroku create b1-chatbot
heroku config:set OPENAI_API_KEY=your_key
git push heroku main
heroku run python manage.py migrate
```

#### AWS EC2

1. Launch EC2 instance (Ubuntu 20.04)
2. Install Python, pip, nginx
3. Clone repository
4. Configure nginx as reverse proxy
5. Use gunicorn as WSGI server
6. Set up SSL with Let's Encrypt

#### Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

Build and run:
```bash
docker build -t b1-chatbot .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key b1-chatbot
```

---

## Advanced Features

### Query Classification System

The chatbot automatically classifies queries into categories (openapp/rag.py:96-105):

```python
MARKET_TOPICS = {
    'transaction_stats': ['sales volume', 'price trends', 'market activity'],
    'home_prices': ['average prices', 'price per sq ft', 'luxury segment pricing'],
    'market_insights': ['investment potential', 'market growth', 'future predictions'],
    'navigation': ['locations', 'communities', 'property types']
}
```

**Benefits:**
- More relevant responses
- Category-specific information enhancement
- Better analytics on user interests

### Response Enhancement

Based on query type, responses are enriched with additional context (openapp/rag.py:107-122):

```python
enhancements = {
    "Property Information": "B1 Properties specializes in ultra-luxury properties...",
    "Market Insights": "As Dubai's premier luxury brokerage...",
    "B1 Properties Services": "Our founder, Babak Jafari...",
    "Transaction History": "Record-breaking AED 128 million villa sale...",
}
```

### Custom Knowledge Base Updates

To update the knowledge base:

1. Edit `openapp/dubai_property.txt`
2. Use `##chunk##` as separator for sections:
   ```
   Section 1 content here
   ##chunk##
   Section 2 content here
   ##chunk##
   Section 3 content here
   ```
3. Restart server to reload embeddings

**Best practices:**
- Keep chunks focused on single topics
- Use clear, descriptive language
- Include relevant keywords
- Update regularly with new information

---

## Troubleshooting

### Common Issues

#### 1. OpenAI API Key Error

**Error:** `openai.AuthenticationError: Incorrect API key provided`

**Solution:**
- Verify API key in `.env` file
- Check for spaces/typos
- Ensure key is active on OpenAI dashboard
- Regenerate key if necessary

#### 2. Rate Limit Exceeded

**Error:** `openai.RateLimitError: Rate limit exceeded`

**Solution:**
- Check your OpenAI usage limits
- Upgrade your OpenAI plan
- Implement request caching
- Add delays between requests

#### 3. Module Import Error

**Error:** `ModuleNotFoundError: No module named 'langchain'`

**Solution:**
```bash
pip install -r requirements.txt --force-reinstall
```

#### 4. Database Migration Error

**Error:** `django.db.migrations.exceptions.InconsistentMigrationHistory`

**Solution:**
```bash
# Delete migrations and database
rm -rf openapp/migrations/
rm db.sqlite3

# Recreate migrations
python manage.py makemigrations openapp
python manage.py migrate
python manage.py createsuperuser
```

#### 5. Static Files Not Loading

**Solution:**
```bash
python manage.py collectstatic --noinput
```

Update `settings.py`:
```python
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
```

#### 6. Chroma DB Persistence Warning

**Warning:** `Chroma is running in ephemeral mode`

**Solution:**
Add persistence in `openapp/rag.py:47`:
```python
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)
```

---

## Testing

### Run Tests

```bash
python manage.py test
```

### Manual Testing Checklist

- [ ] User registration works
- [ ] User login/logout works
- [ ] Chat interface loads correctly
- [ ] Messages are saved to database
- [ ] Chat history displays properly
- [ ] Delete history clears all messages
- [ ] OpenAI API returns responses
- [ ] Error handling works (invalid API key)
- [ ] Telegram bot responds (if configured)

### Load Testing

Use Apache Bench:
```bash
ab -n 100 -c 10 http://127.0.0.1:8000/
```

---

## Performance Optimization

### 1. Database Optimization

**Add indexes:**
```python
class ChatGptBot(models.Model):
    # ... fields ...

    class Meta:
        indexes = [
            models.Index(fields=['user', '-created_at']),
        ]
```

### 2. Caching Responses

Implement Redis caching for common queries:
```python
from django.core.cache import cache

# Check cache first
cached_response = cache.get(f"chat_{user_input}")
if cached_response:
    return cached_response

# Generate response
response = b1_model.get_answer(user_input)

# Cache for 1 hour
cache.set(f"chat_{user_input}", response, 3600)
```

### 3. Async Processing

Convert views to async for better performance:
```python
from django.views.generic import ListView
from asgiref.sync import sync_to_async

class HomeView(LoginRequiredMixin, ListView):
    async def post(self, request, *args, **kwargs):
        # Async processing
        response = await sync_to_async(b1_model.get_answer)(user_input)
```

---

## Security Best Practices

1. **Never commit `.env` file** - Add to `.gitignore`
2. **Use strong SECRET_KEY** - Generate with:
   ```python
   from django.core.management.utils import get_random_secret_key
   print(get_random_secret_key())
   ```
3. **Enable HTTPS** in production
4. **Set DEBUG=False** in production
5. **Implement rate limiting** to prevent abuse
6. **Sanitize user inputs** (Django does this by default)
7. **Use environment variables** for all sensitive data
8. **Regular dependency updates**: `pip install --upgrade -r requirements.txt`

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Coding Standards

- Follow PEP 8 style guide
- Write docstrings for all functions
- Add type hints where applicable
- Write unit tests for new features
- Update README with new features

### Suggested Improvements

- [ ] Add response streaming for real-time updates
- [ ] Implement conversation memory (multi-turn context)
- [ ] Add voice input/output capabilities
- [ ] Create analytics dashboard for admin
- [ ] Multi-language support
- [ ] Export chat history as PDF
- [ ] Add image/document upload for property queries
- [ ] Implement user feedback system (thumbs up/down)
- [ ] Create API endpoints for third-party integrations
- [ ] Add A/B testing for different prompts

---

## FAQ

**Q: How much does it cost to run this chatbot?**
A: Costs depend on usage. OpenAI charges per token. Estimate: ~$0.10-0.50 per 100 conversations with GPT-4.

**Q: Can I use a different LLM instead of OpenAI?**
A: Yes! LangChain supports multiple LLMs (Anthropic Claude, Google PaLM, etc.). Modify `openapp/rag.py`.

**Q: How do I add more knowledge to the chatbot?**
A: Edit `openapp/dubai_property.txt` and add new sections separated by `##chunk##`.

**Q: Can this work for other industries besides real estate?**
A: Absolutely! Replace the knowledge base and customize the prompts for your domain.

**Q: Is the chat data stored securely?**
A: Yes, but ensure you use HTTPS in production and comply with data protection regulations (GDPR, etc.).

**Q: Can multiple users chat simultaneously?**
A: Yes, Django handles concurrent requests. For high traffic, use load balancing.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 B1 Properties

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## Contact

**Project Developer**: [Your Name]
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Portfolio: [your-portfolio.com](https://your-portfolio.com)

**B1 Properties**
- Website: [B1 Properties](https://b1properties.com)
- Founder: Babak Jafari

---

## Acknowledgments

- **OpenAI** for GPT-4 and embeddings API
- **LangChain** for the RAG framework
- **Django Software Foundation** for the web framework
- **Chroma** for the vector database
- **Telegram** for the bot platform
- **Vercel** for hosting platform
- All contributors and testers

---

## Resources & References

### Documentation
- [OpenAI API Docs](https://platform.openai.com/docs)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Django Documentation](https://docs.djangoproject.com/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Telegram Bot API](https://core.telegram.org/bots/api)

### Tutorials
- [Building RAG Applications with LangChain](https://python.langchain.com/docs/use_cases/question_answering/)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
- [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)

### Related Projects
- [LangChain ChatBot Examples](https://github.com/langchain-ai/langchain/tree/master/templates)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)

---

## Support

If you found this project helpful, please:
- ⭐ **Star this repository** on GitHub
- 🐛 **Report bugs** via [GitHub Issues](https://github.com/yourusername/b1_chatbot/issues)
- 💡 **Suggest features** via [Discussions](https://github.com/yourusername/b1_chatbot/discussions)
- 📣 **Share with others** who might find it useful

---

## Changelog

### Version 1.0.0 (Current)
- Initial release
- RAG architecture implementation
- Web interface with Django
- Telegram bot integration
- User authentication system
- Chat history management
- Vercel deployment configuration

### Planned for Version 2.0.0
- Conversation memory (multi-turn context)
- Response streaming
- Analytics dashboard
- Voice input/output
- Multi-language support

---

**Built with ❤️ using Django, OpenAI, and LangChain**

**Powered by RAG Architecture for Accurate, Context-Aware AI Responses**

  
  
  
  "B1 Chatbot - AI-Powered Real Estate Assistant with RAG Architecture"

  Developed an intelligent real estate chatbot for B1 Properties using Django, OpenAI GPT-4, 
  and LangChain's RAG (Retrieval-Augmented Generation) architecture. Implemented semantic 
  search with vector embeddings (Chroma DB) to provide accurate, context-aware responses 
  about Dubai's luxury property market. Features include user authentication, persistent chat
   history, query classification, custom knowledge base integration, and Telegram bot 
  support. Deployed on Vercel with comprehensive error handling and production-ready 
  configuration.

  Key Technologies: Django, Python, OpenAI GPT-4, LangChain, RAG Architecture, Chroma Vector
  Database, OpenAI Embeddings, Telegram Bot API, Vercel

  Advanced Concepts:
  - Retrieval-Augmented Generation (RAG)
  - Vector Embeddings & Semantic Search
  - Natural Language Processing (NLP)
  - Large Language Models (LLMs)
  - Document Chunking & Text Splitting
  - Custom Prompt Engineering

  ---
  Why This Project Stands Out

  1. Modern AI Architecture: Uses cutting-edge RAG technology
  2. Production-Ready: Vercel deployment, error handling, authentication
  3. Real-World Application: Solves actual business needs for luxury real estate
  4. Multi-Platform: Web interface + Telegram integration
  5. Custom Knowledge Base: Not just a generic chatbot - domain-specific intelligence
  6. Scalable Design: Vector database allows easy knowledge expansion

  ---
  Skills Demonstrated

  AI/ML Skills

  - Large Language Model (LLM) integration
  - RAG architecture implementation
  - Vector database management
  - Embedding generation and semantic search
  - Prompt engineering

  Backend Development

  - Django framework
  - RESTful architecture
  - Database modeling (ORM)
  - User authentication & authorization
  - Session management

  Integration & DevOps

  - OpenAI API integration
  - Telegram Bot API
  - Environment configuration
  - Serverless deployment (Vercel)
  - Version control (Git)

  ---
  This is an impressive, production-grade AI project that demonstrates your understanding of
  modern AI architectures, full-stack development, and real-world application deployment.
  Perfect for showcasing advanced technical skills on your resume!


# b1_chatbot
The code is written in Python, using the Django web framework and the OpenAI API for natural language processing. The repository includes all the necessary files to run the application, as well as documentation on how to set it up and use it.

If you're interested in exploring the possibilities of natural language generation with Django and the OpenAI API, this repository is a great starting point. Feel free to clone the repo and start experimenting!

Prerequisites:

Python >=3.10 


Git 


OpenAI API key 


Telegram bot token (if using Telegram integration)

Clone the repository 
Create Venv and Install dependencies by running : pip install -r requirements.txt 
Run the migrations: 
python manage.py makemigrations 
python manage.py migrate

