import os
import re
import json
import warnings
import requests
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_google_genai import ChatGoogleGenerativeAI

class VisitorCounter:
    def __init__(self, counter_file="visitor_count.json"):
        self.counter_file = counter_file
        self.data = self.load_counter()
    
    def load_counter(self):
        """Load visitor count from file"""
        try:
            if os.path.exists(self.counter_file):
                with open(self.counter_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        # Default data structure
        return {
            'total_visits': 0,
            'first_visit': None,
            'last_visit': None,
            'daily_visits': {}
        }
    
    def save_counter(self):
        """Save visitor count to file"""
        try:
            with open(self.counter_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            st.error(f"Error saving counter: {e}")
    
    def increment_visit(self):
        """Increment visitor counter"""
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        
        # Increment total visits
        self.data['total_visits'] += 1
        
        # Update timestamps
        if not self.data['first_visit']:
            self.data['first_visit'] = now.isoformat()
        self.data['last_visit'] = now.isoformat()
        
        # Update daily visits
        if today not in self.data['daily_visits']:
            self.data['daily_visits'][today] = 0
        self.data['daily_visits'][today] += 1
        
        
        self.save_counter()
    
    def get_stats(self):
        """Get visitor statistics"""
        today = datetime.now().strftime('%Y-%m-%d')
        today_visits = self.data['daily_visits'].get(today, 0)
        
        return {
            'total_visits': self.data['total_visits'],
            'today_visits': today_visits,
            'first_visit': self.data['first_visit'],
            'last_visit': self.data['last_visit']
        }

def display_visitor_counter():
    """Display visitor counter in sidebar"""
    # Initialize counter
    if 'counter' not in st.session_state:
        st.session_state.counter = VisitorCounter()
        # Increment on first load of session
        st.session_state.counter.increment_visit()
    
    counter = st.session_state.counter
    stats = counter.get_stats()
    
    # Display counter in sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Visitor Stats")
        
        # Main metrics
        st.metric("👥 Total Visits", stats['total_visits'])
        st.metric("📅 Today's Visits", stats['today_visits'])
        
        # Show as a nice badge
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            margin: 10px 0;
        ">
            <h3 style="margin: 0; color: white;"> Visitor #{stats['total_visits']}</h3>
            <p style="margin: 5px 0 0 0; color: white;">Thanks for visiting!</p>
        </div>
        """, unsafe_allow_html=True)

def display_detailed_stats():
    """Display detailed visitor statistics"""
    if 'counter' not in st.session_state:
        return
    
    counter = st.session_state.counter
    stats = counter.get_stats()
    
    st.subheader("📈 Detailed Visitor Analytics")
    
    # Overview cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Visits", stats['total_visits'])

    with col2:
        st.metric("📅 Today's Visits", stats['today_visits'])
    
    with col3:
        if stats['total_visits'] > 0:
            avg_daily = stats['total_visits'] / max(1, len(counter.data['daily_visits']))
            st.metric("📊 Daily Average", f"{avg_daily:.1f}")
    
    # Timeline info
    if stats['first_visit']:
        first_date = datetime.fromisoformat(stats['first_visit']).strftime('%Y-%m-%d %H:%M')
        last_date = datetime.fromisoformat(stats['last_visit']).strftime('%Y-%m-%d %H:%M')
        st.info(f"🎯 **First visitor:** {first_date} • **Latest visitor:** {last_date}")

class NewsletterSettings:
    def __init__(self):
        self.default_settings = {
            # AI Settings
            'ai_temperature': 0.3,
            'ai_model': 'gemini-2.0-flash',
            'max_articles': 3,
            'content_length': 'medium',
            
            # Content Settings
            'writing_tone': 'casual',
            'target_audience': 'general',
            'include_links': True,
            'include_quotes': True,
            'include_emojis': True,
            
            # Quality Settings
            'min_word_count': 800,
            'max_word_count': 1500,
            'readability_target': 'intermediate',
            'section_count': 4,
            'intro_style': 'personal'
        }
    
    def get_setting(self, key):
        if 'newsletter_settings' not in st.session_state:
            st.session_state.newsletter_settings = self.default_settings.copy()
        return st.session_state.newsletter_settings.get(key, self.default_settings[key])
    
    def set_setting(self, key, value):
        if 'newsletter_settings' not in st.session_state:
            st.session_state.newsletter_settings = self.default_settings.copy()
        st.session_state.newsletter_settings[key] = value
    
    def reset_to_defaults(self):
        st.session_state.newsletter_settings = self.default_settings.copy()

def analyze_newsletter(newsletter_content):
    """Analyze newsletter content and return metrics"""
    if not newsletter_content:
        return {}
    
    # Basic metrics
    word_count = len(newsletter_content.split())
    char_count = len(newsletter_content)
    sentence_count = len(re.findall(r'[.!?]+', newsletter_content))
    paragraph_count = len([p for p in newsletter_content.split('\n\n') if p.strip()])
    
    # Reading metrics
    reading_time = max(1, round(word_count / 200))
    
    # Simple readability score (approximation)
    avg_sentence_length = word_count / max(1, sentence_count)
    readability_score = max(0, min(100, 100 - (avg_sentence_length - 15) * 2))
    
    # Content features
    bullet_points = newsletter_content.count('•') + newsletter_content.count('*')
    links_count = newsletter_content.count('http')
    emojis_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', newsletter_content))
    
    # Engagement score (simple algorithm)
    engagement_score = min(100, max(0, 
        (readability_score * 0.3) + 
        (min(bullet_points * 5, 25)) + 
        (min(links_count * 10, 20)) + 
        (min(emojis_count * 2, 15)) +
        (30 if 800 <= word_count <= 1500 else 10)
    ))
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'reading_time': reading_time,
        'readability_score': readability_score,
        'bullet_points': bullet_points,
        'links_count': links_count,
        'emojis_count': emojis_count,
        'engagement_score': engagement_score
    }

def create_sidebar_analytics(analytics):
    """Create analytics dashboard in sidebar"""
    if not analytics:
        return
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Newsletter Analytics")
    
    # Key metrics
    st.sidebar.metric("📝 Words", analytics['word_count'])
    st.sidebar.metric("⏱️ Read Time", f"{analytics['reading_time']} min")
    st.sidebar.metric("📖 Readability", f"{analytics['readability_score']:.0f}/100")
    st.sidebar.metric("🎯 Engagement", f"{analytics['engagement_score']:.0f}/100")
    
    # Compact chart
    if st.sidebar.checkbox("📈 Show Details"):
        # Mini engagement gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = analytics['engagement_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Engagement Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
        st.sidebar.plotly_chart(fig_gauge, use_container_width=True)
        
        # Content breakdown
        st.sidebar.write("**Content Elements:**")
        st.sidebar.write(f"• Sentences: {analytics['sentence_count']}")
        st.sidebar.write(f"• Paragraphs: {analytics['paragraph_count']}")
        st.sidebar.write(f"• Bullet points: {analytics['bullet_points']}")
        st.sidebar.write(f"• Links: {analytics['links_count']}")
        st.sidebar.write(f"• Emojis: {analytics['emojis_count']}")

def validate_google_api_key(api_key):
    try:
        llm = ChatGoogleGenerativeAI(
            temperature=0.3, 
            model="gemini-2.0-flash", 
            google_api_key=api_key
        )
        test_response = llm.invoke("Hello")
        return True, "✅ Google API key is valid!"
    except Exception as e:
        return False, f"❌ Invalid Google API key: {str(e)}"

def validate_serper_api_key(api_key):
    try:
        search = GoogleSerperAPIWrapper(k=1, type="search", serper_api_key=api_key)
        search.results("test")
        return True, "✅ SERPER API key is valid!"
    except Exception as e:
        return False, f"❌ Invalid SERPER API key: {str(e)}"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def search_serpapi(query, serper_api_key):
    search = GoogleSerperAPIWrapper(k=5, type="search", serper_api_key=serper_api_key)
    response_json = search.results(query)
    return response_json

def safe_parser(urls):
    try:
        return json.loads(urls)
    except json.JSONDecodeError:
        url = re.findall(r'(https?://\S+)', urls)
        return url if url else ['www.google.com']

def choose_best_article(response_json, query, google_api_key, settings):
    response_str = json.dumps(response_json)
    
    # Use settings for AI parameters
    llm = ChatGoogleGenerativeAI(
        temperature=settings.get_setting('ai_temperature'), 
        model=settings.get_setting('ai_model'), 
        google_api_key=google_api_key
    )
    
    template = """You are a world class journalist, researcher, writer, software developer, fact checker and online media expert.
    You are best at finding the most relevant, interesting and useful articles in certain topics.
    
    Query response: {response_str}
    
    Above are the search result for the query: {query}
    
    Please choose the best 3 articles from the list and return ONLY an array or list of the URLs.
    Do not include anything else -
    return ONLY an array of the URLs follow this strictly.
    Return only the 3 best, valid, recent article URLs from the list.
    Do not add text, explanations, or placeholders. 
    If fewer than 3 exist, just return however many exist.
    
    Format: ["url1", "url2", "url3"]
    """
    
    prompt_template = PromptTemplate(
        input_variables=["response_str", "query"],
        template=template
    )
    
    article_choose_chain = prompt_template | llm | StrOutputParser()
    urls = article_choose_chain.invoke({"response_str": response_str, "query": query})
    
    return safe_parser(urls)

def extract_and_split(urls):
    try:
        loader = UnstructuredURLLoader(urls=urls)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = text_splitter.split_documents(documents)
        db = FAISS.from_documents(docs, get_embeddings())
        return db
    except Exception as e:
        st.error(f"Error extracting content from URLs: {str(e)}")
        return None

def summarizer(db, query, google_api_key, settings, k=4):
    docs = db.similarity_search(query, k=k)
    docs_content = " ".join([doc.page_content for doc in docs])
    
    llm = ChatGoogleGenerativeAI(
        temperature=settings.get_setting('ai_temperature'), 
        model=settings.get_setting('ai_model'), 
        google_api_key=google_api_key
    )
    
    template2 = """
    {docs}
    As a world class journalist, researcher, writer, software developer, newsletter creator and blogger you will summarize the above text into a concise order to create a newsletter around {query}.
    This newsletter is going to be read by thousands of people so make sure it is engaging, interesting and informative. Also this will be sent as email. The format should be like Tim Ferriss's 5-Bullet Friday newsletter.
    
    Please follow all of the following guidelines:
    1. Make sure the content is engaging, interesting and informative.
    2. Make sure the content is not too long, It should be concise and to the point.
    3. The content should address the {query} topic very well.
    4. The content should be in the format of a newsletter.
    5. The content should be in English and easy to understand.
    6. The content should have a good flow and structure.
    7. The content should give the audience actionable insights.
    8. And at the end, dont includename [Your Name] tag.
    
    SUMMARY:
    """
    
    prompt_template2 = PromptTemplate(
        input_variables=["docs", "query"],
        template=template2
    )
    
    summarizer_chain = prompt_template2 | llm | StrOutputParser()
    response = summarizer_chain.invoke({'docs': docs_content, 'query': query})
    
    return response.replace('\n', "")

def create_enhanced_newsletter(summary, query, google_api_key, author_name, settings):
    """Enhanced newsletter creation with advanced settings"""
    summaries = str(summary)
    
    # Get settings
    tone = settings.get_setting('writing_tone')
    audience = settings.get_setting('target_audience')
    content_length = settings.get_setting('content_length')
    include_links = settings.get_setting('include_links')
    include_quotes = settings.get_setting('include_quotes')
    include_emojis = settings.get_setting('include_emojis')
    intro_style = settings.get_setting('intro_style')
    section_count = settings.get_setting('section_count')
    min_words = settings.get_setting('min_word_count')
    max_words = settings.get_setting('max_word_count')
    
    # AI model with settings
    llm = ChatGoogleGenerativeAI(
        temperature=settings.get_setting('ai_temperature'), 
        model=settings.get_setting('ai_model'), 
        google_api_key=google_api_key
    )
    
    # Tone descriptions
    tone_descriptions = {
        'casual': 'casual, friendly, and conversational',
        'professional': 'professional, polished, and authoritative',
        'technical': 'technical, detailed, and precise',
        'inspirational': 'inspiring, motivating, and uplifting',
        'humorous': 'light-hearted, engaging, and occasionally humorous'
    }
    
    # Audience descriptions
    audience_descriptions = {
        'general': 'general audience with varied backgrounds',
        'professionals': 'business professionals and executives',
        'technical': 'technical experts and specialists',
        'students': 'students and lifelong learners',
        'entrepreneurs': 'entrepreneurs and startup founders'
    }
    
    # Length mapping
    length_mapping = {
        'short': '500-800 words',
        'medium': '800-1200 words',
        'long': '1200-2000 words'
    }
    
    # Enhanced template with settings
    template3 = f"""
    {{summaries}}
        As a world class journalist, researcher, article, newsletter and blog writer, 
        you'll use the text above as the context about {{query}}
        to write an excellent newsletter to be sent to subscribers about {{query}}.
        
        This newsletter will be sent as an email. The format is going to be like
        Tim Ferriss' "5-Bullet Friday" newsletter.
        
        ADVANCED WRITING REQUIREMENTS:
        
        Writing Style: Write in a {tone_descriptions.get(tone, 'casual')} tone
        Target Audience: {audience_descriptions.get(audience, 'general audience')}
        Content Length: Aim for {length_mapping.get(content_length, '800-1200 words')} ({min_words}-{max_words} words)
        Number of Main Sections: Organize content into {section_count} main sections
        Introduction Style: Use a {intro_style} introduction approach
        
        Content Requirements:
        - {'Include relevant external links throughout the content' if include_links else 'Focus on content without external links'}
        - {'End with an inspirational or thought-provoking quote' if include_quotes else 'End with actionable insights instead of quotes'}
        - {'Use emojis appropriately to enhance readability' if include_emojis else 'Use minimal or no emojis, focus on text clarity'}
        
        Make sure to write it informally - no "Dear" or any other formalities. Start the newsletter with
        `Hi All!
          Here is your weekly dose of the newsletter, a list of what I find interesting
          and worth exploring.`
          
        Make sure to also write a backstory about the topic - make it personal, engaging and lighthearted before
        going into the meat of the newsletter.
        
        Please follow all of the following guidelines:
        1/ Make sure the content is engaging, informative with good data
        2/ Make sure the content matches the specified length target
        3/ The content should address the {{query}} topic very well
        4/ The content needs to be good and informative
        5/ The content needs to be written in a way that is easy to read, digest and understand
        6/ The content needs to give the audience actionable advice & insights including resources and links if necessary.
        
        If there are books, or products involved, make sure to add amazon links to the products.
        
        As a signoff, write a clever quote related to learning, general wisdom, living a good life. Be creative with this one - and then,
        Sign with "{{author_name}} 
          - Innovation Learner"
        
        Please ensure the newsletter meets these specific requirements while maintaining high quality and engagement.
        
        NEWSLETTER-->:
    """
    
    prompt_template3 = PromptTemplate(
        input_variables=["summaries", "query", "author_name"],
        template=template3
    )   
    
    newsletter_chain = prompt_template3 | llm | StrOutputParser()
    response = newsletter_chain.invoke({'summaries': summaries, 'query': query, 'author_name': author_name})
    return response

def display_advanced_settings():
    """Display advanced settings interface"""
    st.subheader("🔧 Advanced Settings")
    
    settings = NewsletterSettings()
    
    # Create tabs for different setting categories
    tab1, tab2, tab3 = st.tabs(["🤖 AI Settings", "📝 Content Style", "📊 Quality"])
    
    with tab1:
        st.write("**AI Configuration**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # AI Temperature
            temperature = st.slider(
                "🌡️ AI Creativity:",
                min_value=0.0,
                max_value=1.0,
                value=settings.get_setting('ai_temperature'),
                step=0.1,
                help="Higher = more creative, Lower = more focused"
            )
            settings.set_setting('ai_temperature', temperature)
            
            # Model selection
            model_options = {
                'gemini-2.0-flash': 'Gemini 2.0 Flash (Fast)',
                'gemini-1.5-pro': 'Gemini 1.5 Pro (Advanced)',
                'gemini-1.0-pro': 'Gemini 1.0 Pro (Stable)',
                'gemini-2.0-flash-lite': 'Gemini 2.0 Flash-Lite',
                'gemini-2.5-flash-lite': 'Gemini 2.5 Flash-Lite',
                'Gemini-2.5-flash': 'Gemini 2.5 Flash',
                'Gemini-2.5-pro': 'Gemini 2.5 Pro'
            }
            
            selected_model = st.selectbox(
                "🧠 AI Model:",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                index=list(model_options.keys()).index(settings.get_setting('ai_model'))
            )
            settings.set_setting('ai_model', selected_model)
        
        with col2:
            # Max articles
            max_articles = st.slider(
                "📄 Articles to Analyze:",
                min_value=1,
                max_value=10,
                value=settings.get_setting('max_articles'),
                help="More articles = better insights but slower"
            )
            settings.set_setting('max_articles', max_articles)
            
            # Content length
            length_options = {
                'short': 'Short (500-800 words)',
                'medium': 'Medium (800-1200 words)', 
                'long': 'Long (1200-2000 words)'
            }
            
            content_length = st.selectbox(
                "📏 Newsletter Length:",
                options=list(length_options.keys()),
                format_func=lambda x: length_options[x],
                index=list(length_options.keys()).index(settings.get_setting('content_length'))
            )
            settings.set_setting('content_length', content_length)
    
    with tab2:
        st.write("**Content & Style**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Writing tone
            tone_options = {
                'casual': 'Casual & Friendly',
                'professional': 'Professional',
                'technical': 'Technical & Detailed',
                'inspirational': 'Inspirational',
                'humorous': 'Light & Humorous'
            }
            
            writing_tone = st.selectbox(
                "🎭 Writing Tone:",
                options=list(tone_options.keys()),
                format_func=lambda x: tone_options[x],
                index=list(tone_options.keys()).index(settings.get_setting('writing_tone'))
            )
            settings.set_setting('writing_tone', writing_tone)
            
            # Target audience
            audience_options = {
                'general': 'General Audience',
                'professionals': 'Business Professionals',
                'technical': 'Technical Experts',
                'students': 'Students & Learners',
                'entrepreneurs': 'Entrepreneurs'
            }
            
            target_audience = st.selectbox(
                "👥 Target Audience:",
                options=list(audience_options.keys()),
                format_func=lambda x: audience_options[x],
                index=list(audience_options.keys()).index(settings.get_setting('target_audience'))
            )
            settings.set_setting('target_audience', target_audience)
        
        with col2:
            # Content features
            st.write("**Include in Newsletter:**")
            
            include_links = st.checkbox(
                "🔗 External Links",
                value=settings.get_setting('include_links')
            )
            settings.set_setting('include_links', include_links)
            
            include_quotes = st.checkbox(
                "💬 Inspirational Quotes",
                value=settings.get_setting('include_quotes')
            )
            settings.set_setting('include_quotes', include_quotes)
            
            include_emojis = st.checkbox(
                "😊 Emojis",
                value=settings.get_setting('include_emojis')
            )
            settings.set_setting('include_emojis', include_emojis)
    
    with tab3:
        st.write("**Quality & Structure**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Word count targets
            min_words = st.number_input(
                "📝 Minimum Words:",
                min_value=200,
                max_value=2000,
                value=settings.get_setting('min_word_count'),
                step=100
            )
            settings.set_setting('min_word_count', min_words)
            
            max_words = st.number_input(
                "📝 Maximum Words:",
                min_value=min_words,
                max_value=5000,
                value=max(settings.get_setting('max_word_count'), min_words),
                step=100
            )
            settings.set_setting('max_word_count', max_words)
        
        with col2:
            # Newsletter structure
            intro_options = {
                'personal': 'Personal & Conversational',
                'professional': 'Professional Introduction',
                'story': 'Story-based Opening',
                'direct': 'Direct & To-the-point'
            }
            
            intro_style = st.selectbox(
                "📖 Introduction Style:",
                options=list(intro_options.keys()),
                format_func=lambda x: intro_options[x],
                index=list(intro_options.keys()).index(settings.get_setting('intro_style'))
            )
            settings.set_setting('intro_style', intro_style)
            
            section_count = st.slider(
                "📋 Main Sections:",
                min_value=2,
                max_value=8,
                value=settings.get_setting('section_count'),
                help="Number of main content sections"
            )
            settings.set_setting('section_count', section_count)
    
    # Settings actions
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reset to Defaults"):
            settings.reset_to_defaults()
            st.success("Settings reset!")
            st.rerun()
    
    with col2:
        current_settings = st.session_state.get('newsletter_settings', settings.default_settings)
        settings_json = json.dumps(current_settings, indent=2)
        
        st.download_button(
            "📥 Export Settings",
            settings_json,
            file_name="newsletter_settings.json",
            mime="application/json"
        )
