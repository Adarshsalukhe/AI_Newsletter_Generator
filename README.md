# AI_Newsletter_Generator

# APP Link
Link - https://ai-newsletter-generator.streamlit.app/

# 📖 Description

The AI Newsletter Generator allows you to instantly generate high-quality newsletters tailored to your preferences with just a single input. Whether you need a Professional, Technical, or Casual tone, the app leverages state-of-the-art AI models to craft content that suits your needs.

You can customize the generation process by:

Selecting your preferred Gemini model<br>
Adjusting the temperature (creativity level)<br>
Choosing whether to include external links, emojis, or maintain a strict professional style<br>

Additional functionality includes the ability to:
Regenerate newsletters for alternative drafts<br>
Download newsletters in one click<br>
Copy text directly for quick usage<br>
Retrieve the top 3 source links that inspired the newsletter content<br>

The project integrates Google API and SERPER API for real-time data access. A detailed guide is provided for obtaining and configuring your personal API keys.

# 🚀 Features

🔑 API integration with Google & SERPER<br>
📝 Generate Professional / Technical / Casual newsletters<br>
🎛️ Control model selection, temperature, style, links, and emoji usage<br>
🔄 Regenerate, download, or copy newsletter content<br>
🔗 Display top source links for reference<br>
📊 Visualize trends and topics with Plotly & Matplotlib<br>
🌐 Built with LangChain and LangGraph for modular AI pipelines<br> 

# 🛠️ Tech Stack
AI & LLM:<br>
Google Gemini (Large Language Model)<br>
HuggingFace Embeddings<br>
LangChain (LLM orchestration)<br>
LangGraph (workflow graphing for LLM apps)<br>

Framework & Tools:<br>
Streamlit (interactive web app framework)<br>
Plotly (interactive visualizations)<br>
Matplotlib (data visualization)<br>
Pandas & NumPy (data handling and processing)<br>
Requests / HTTPx (API communication)<br>

Programming & Environment:<br>
Python (primary language)<br>
Jupyter / VS Code (development & prototyping)<br>
Git & GitHub (version control & collaboration)<br>
FAISS Vectorstore for database <br>

Infrastructure & Deployment:<br>
Windows 11 (development OS)<br>
Docker (optional containerized deployment)<br>
Streamlit Cloud / Hugging Face Spaces (hosting options)<br>

# Installation:
1. Clone the repository<br>
git clone https://github.com/Adarshsalukhe/AI_Newsletter_Generator.git<br>
cd AI_Newsletter_Generator<br>

2. (Optional but recommended) Create and activate a virtual environment<br>
python -m venv venv<br>
#On macOS/Linux:<br>
source venv/bin/activate<br>
#On Windows (PowerShell):<br>
.\venv\Scripts\Activate.ps1<br>
#Or on Windows (Command Prompt):<br>
venv\Scripts\activate<br>

3. Install Python dependencies<br>
pip install -r requirements.txt<br>

# Project Structure:

AI-newsletter-generator/<br>
│── helper.py     # Source code<br>
│── app.py        # UI\UX<br>
|── requirements.txt  # install packages <br>
│── README.md     # Project documentation<br>

# Common Issues:<br>

API Key Errors:<br>
❌ Error: Invalid API Key or API request failed<br>
✅ Fix: Make sure you’ve added your Google API Key and SERPER API Key correctly in the given field. <br>
Double-check for extra spaces or quotes.<br>

Missing Dependencies:<br>
❌ Error: ModuleNotFoundError: No module named 'xyz'<br>
✅ Fix: Run<br>
pip install -r requirements.txt <br>
to install all required dependencies.<br>

Streamlit App Not Launching:<br>
❌ Error: Nothing happens after running streamlit run app.py<br>
✅ Fix: Ensure you are inside the correct project directory and your virtual environment is activated.<br>

Version Conflicts:<br>
❌ Error: Dependency version mismatch (e.g., with langchain, streamlit, or plotly).<br>
✅ Fix: Upgrade/downgrade packages as specified in requirements.txt, or create a fresh virtual.<br>

# 👤 Authors / Acknowledgements:<br>
Your Name – @AdarshSalukhe<br>




