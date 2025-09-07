# AI_Newsletter_Generator

# APP Link
Link - https://ai-newsletter-generator.streamlit.app/

# 📖 Description

The AI Newsletter Generator allows you to instantly generate high-quality newsletters tailored to your preferences with just a single input. Whether you need a Professional, Technical, or Casual tone, the app leverages state-of-the-art AI models to craft content that suits your needs.

You can customize the generation process by:

Selecting your preferred Gemini model
Adjusting the temperature (creativity level)
Choosing whether to include external links, emojis, or maintain a strict professional style

Additional functionality includes the ability to:
Regenerate newsletters for alternative drafts
Download newsletters in one click
Copy text directly for quick usage
Retrieve the top 3 source links that inspired the newsletter content

The project integrates Google API and SERPER API for real-time data access. A detailed guide is provided for obtaining and configuring your personal API keys.

# 🚀 Features

🔑 API integration with Google & SERPER
📝 Generate Professional / Technical / Casual newsletters
🎛️ Control model selection, temperature, style, links, and emoji usage
🔄 Regenerate, download, or copy newsletter content
🔗 Display top source links for reference
📊 Visualize trends and topics with Plotly & Matplotlib
🌐 Built with LangChain and LangGraph for modular AI pipelines 

# 🛠️ Tech Stack
AI & LLM:
Google Gemini (Large Language Model)
HuggingFace Embeddings
LangChain (LLM orchestration)
LangGraph (workflow graphing for LLM apps)

Framework & Tools:
Streamlit (interactive web app framework)
Plotly (interactive visualizations)
Matplotlib (data visualization)
Pandas & NumPy (data handling and processing)
Requests / HTTPx (API communication)

Programming & Environment:
Python (primary language)
Jupyter / VS Code (development & prototyping)
Git & GitHub (version control & collaboration)
FAISS Vectorstore for database 

Infrastructure & Deployment:
Windows 11 (development OS)
Docker (optional containerized deployment)
Streamlit Cloud / Hugging Face Spaces (hosting options)

# Installation:
# 1. Clone the repository
git clone https://github.com/Adarshsalukhe/AI_Newsletter_Generator.git
cd AI_Newsletter_Generator

# 2. (Optional but recommended) Create and activate a virtual environment
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Or on Windows (Command Prompt):
venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# Project Structure:

AI-newsletter-generator/
│── helper.py     # Source code
│── app.py        # UI\UX
|── requirements.txt  # install packages 
│── README.md     # Project documentation

# Common Issues:

API Key Errors
❌ Error: Invalid API Key or API request failed
✅ Fix: Make sure you’ve added your Google API Key and SERPER API Key correctly in the given field. 
Double-check for extra spaces or quotes.

Missing Dependencies
❌ Error: ModuleNotFoundError: No module named 'xyz'
✅ Fix: Run
pip install -r requirements.txt 
to install all required dependencies.

Streamlit App Not Launching
❌ Error: Nothing happens after running streamlit run app.py
✅ Fix: Ensure you are inside the correct project directory and your virtual environment is activated.

Version Conflicts
❌ Error: Dependency version mismatch (e.g., with langchain, streamlit, or plotly).
✅ Fix: Upgrade/downgrade packages as specified in requirements.txt, or create a fresh virtual

# 👤 Authors / Acknowledgements
Your Name – @AdarshSalukhe




