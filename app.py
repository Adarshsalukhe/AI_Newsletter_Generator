import streamlit as st
from helper import *
from datetime import datetime

def main():
    st.set_page_config(page_title="AI Newsletter Generator", page_icon="📰", layout="wide")
    
    st.title("🚀 AI Newsletter Generator")
    st.write("Generate professional newsletters with advanced AI controls - Using YOUR API keys")
    
    # Initialize settings
    settings = NewsletterSettings()
    
    display_visitor_counter()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🔑 Enter Your API Keys")
        
        google_api_key = st.text_input(
            "Google AI API Key:",
            type="password",
            help="Get free key from: https://aistudio.google.com/app/apikey"
        )
        
        serper_api_key = st.text_input(
            "SERPER API Key:",
            type="password",
            help="Get free key from: https://serper.dev/"
        )
        
        # Validate API keys
        google_valid = False
        serper_valid = False
        
        if google_api_key:
            google_valid, google_msg = validate_google_api_key(google_api_key)
            if google_valid:
                st.success(google_msg)
            else:
                st.error(google_msg)
        
        if serper_api_key:
            serper_valid, serper_msg = validate_serper_api_key(serper_api_key)
            if serper_valid:
                st.success(serper_msg)
            else:
                st.error(serper_msg)
        
        # Settings and controls (only show if API keys valid)
        if google_valid and serper_valid:
            st.markdown("---")
            st.subheader("⚙️ Settings")
            
            author_name = st.text_input("Author Name:", value="Adarsh")
            
            # Advanced Settings Toggle
            if st.button("🔧 Advanced Settings"):
                st.session_state.show_settings = not st.session_state.get('show_settings', False)
            
            # Current settings summary
            with st.expander("📋 Current Settings"):
                st.write(f"**AI Model:** {settings.get_setting('ai_model')}")
                st.write(f"**Tone:** {settings.get_setting('writing_tone').title()}")
                st.write(f"**Length:** {settings.get_setting('content_length').title()}")
                st.write(f"**Temperature:** {settings.get_setting('ai_temperature')}")
        
        # Analytics will be shown here after generation
        if 'current_analytics' in st.session_state:
            create_sidebar_analytics(st.session_state.current_analytics)
    
    # Main content area
    if not (google_valid and serper_valid):
        st.warning("⚠️ Please provide valid API keys to continue")
        
        st.subheader("🔗 Get Your Free API Keys:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Google AI API Key (FREE)**")
            st.write("1. Go to: https://aistudio.google.com/app/apikey")
            st.write("2. Sign in with Google")
            st.write("3. Create API Key")
            st.write("4. Paste in sidebar")
        
        with col2:
            st.write("**SERPER API Key (FREE)**")
            st.write("1. Go to: https://serper.dev/")
            st.write("2. Sign up free")
            st.write("3. Get API key")
            st.write("4. Paste in sidebar")
        
        st.info("💡 Both services offer generous free tiers!")
    
    else:
        # Show advanced settings if toggled
        if st.session_state.get('show_settings', False):
            display_advanced_settings()
            st.markdown("---")
        
        # Main newsletter generation
        st.success("🎉 API keys validated! Ready to generate newsletters.")
        
        # Topic input
        query = st.text_input(
            "What topic would you like to create a newsletter about?",
            placeholder="e.g., Latest AI developments, Cryptocurrency trends, Space exploration...",
            help="Be specific for better results"
        )
        
        # Generate button
        if st.button("🚀 Generate Newsletter", disabled=not query, type="primary"):
            if not query:
                st.error("Please enter a topic")
            else:
                try:
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Search
                    status_text.text("🔍 Searching for articles...")
                    progress_bar.progress(20)
                    search_results = search_serpapi(query, serper_api_key)
                    
                    # Step 2: Choose articles
                    status_text.text("🎯 Selecting best articles...")
                    progress_bar.progress(40)
                    best_urls = choose_best_article(search_results, query, google_api_key, settings)
                    
                    # Step 3: Extract content
                    status_text.text("📚 Extracting content...")
                    progress_bar.progress(60)
                    db = extract_and_split(best_urls)
                    
                    if db is None:
                        st.error("Failed to extract content. Try a different topic.")
                        return
                    
                    # Step 4: Summarize
                    status_text.text("📋 Summarizing...")
                    progress_bar.progress(80)
                    summary = summarizer(db, query, google_api_key, settings)
                    
                    # Step 5: Create enhanced newsletter
                    status_text.text("✍️ Creating newsletter with your settings...")
                    progress_bar.progress(100)
                    newsletter = create_enhanced_newsletter(summary, query, google_api_key, author_name, settings)
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Analyze newsletter and store in session state
                    analytics = analyze_newsletter(newsletter)
                    st.session_state.current_analytics = analytics
                    
                    # Show results
                    st.success("Newsletter generated with your advanced settings! 🎉")
                    
                    # Settings summary
                    st.info(f"📋 Generated with: {settings.get_setting('writing_tone').title()} tone, "
                           f"{settings.get_setting('content_length')} length, "
                           f"for {settings.get_setting('target_audience')} audience")
                    
                    st.subheader("📰 Your Newsletter")
                    st.markdown(newsletter)
                    
                    # Download and options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            "📥 Download Newsletter",
                            newsletter,
                            file_name=f"newsletter_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt"
                        )
                    
                    with col2:
                        # Copy to clipboard helper
                        if st.button("📋 Copy Text"):
                            st.code(newsletter)
                            st.info("Newsletter text shown above - select all and copy!")
                    
                    with col3:
                        if st.button("🔄 Regenerate"):
                            st.rerun()
                    
                    # Sources
                    with st.expander("📖 Sources Used"):
                        for i, url in enumerate(best_urls, 1):
                            st.write(f"{i}. {url}")
                    
                    # Quick settings adjustment
                    with st.expander("⚙️ Quick Settings Adjustment"):
                        st.write("Not happy with the result? Try adjusting these settings:")
                        
                        quick_col1, quick_col2 = st.columns(2)
                        
                        with quick_col1:
                            if st.button("🎨 More Creative"):
                                settings.set_setting('ai_temperature', min(1.0, settings.get_setting('ai_temperature') + 0.2))
                                st.success("Increased creativity! Regenerate to see changes.")
                            
                            if st.button("📏 Make Longer"):
                                current = settings.get_setting('content_length')
                                if current == 'short': settings.set_setting('content_length', 'medium')
                                elif current == 'medium': settings.set_setting('content_length', 'long')
                                st.success("Increased length! Regenerate to see changes.")
                        
                        with quick_col2:
                            if st.button("🎯 More Focused"):
                                settings.set_setting('ai_temperature', max(0.0, settings.get_setting('ai_temperature') - 0.2))
                                st.success("Increased focus! Regenerate to see changes.")
                            
                            if st.button("📝 Make Shorter"):
                                current = settings.get_setting('content_length')
                                if current == 'long': settings.set_setting('content_length', 'medium')
                                elif current == 'medium': settings.set_setting('content_length', 'short')
                                st.success("Decreased length! Regenerate to see changes.")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("💡 Try adjusting your settings or using a different topic")

if __name__ == "__main__":
    main()