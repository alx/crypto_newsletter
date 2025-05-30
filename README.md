![alt ai](https://github.com/alx/crypto_nesletter/blob/main/assets/header.png?raw=true)

# AI Newsletter Generator

## Description

The AI Newsletter Generator is a Python Streamlit app that automates AI-focused newsletter creation and distribution. It uses a multi-agent system powered by Google Gemini, ingests data from LinkAce, researches with the Exa API, and publishes to WhatsApp via the Waha API.

## Features

  * **Automated Newsletter Creation**: Generates newsletters on specified AI topics.
  * **Multi-Agent Pipeline**: Employs specialized AI agents for:
      * Link Ingestion (LinkAce)
      * Web Research (Exa)
      * Insight Generation
      * Content Writing
      * Editing & Refinement
      * WhatsApp Publishing (Waha)
  * **Google Gemini & Exa API Integration**: Uses Gemini for text generation and Exa for web searches.
  * **Optional LinkAce & WhatsApp Integration**: Fetches links from LinkAce and publishes via Waha.
  * **Interactive UI & Customization**: Streamlit UI, selectable Gemini models, and viewable intermediate agent results.
  * **Markdown Output**: Provides newsletters in downloadable Markdown.

## Setup and Installation

1.  **Clone Repository**:

    ```bash
    git clone <your-repository-url> && cd <repository-name>
    ```

2.  **Create & Activate Virtual Environment**:

    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
    ```

3.  **Install Dependencies**:
    Ensure `requirements.txt` is present (listing `streamlit`, `google-generativeai`, `exa-py`, `requests`, `python-dotenv`, `pydantic`, `karo`).

    ```bash
    pip install -r requirements.txt
    ```

    (Note: Ensure local/private `karo` framework is correctly installed.)

4.  **Configure Environment Variables**:
    Create a `.env` file in the project root as detailed in the [Configuration](#configuration) section.

## Configuration

Create a `.env` file in the project root with the following (also configurable in Streamlit sidebar):

```env
# Core API Keys (Required)
GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
EXA_API_KEY="YOUR_EXA_API_KEY"

# LinkAce Configuration (Optional)
LINKACE_URL="[https://your-linkace-instance.com](https://your-linkace-instance.com)"
LINKACE_TOKEN="YOUR_LINKACE_API_TOKEN"
LINKACE_LIST_ID="YOUR_NUMERIC_LINKACE_LIST_ID"

# Waha (WhatsApp API) Configuration (Optional)
WAHA_API_HOST="http://your-waha-host:port"
WAHA_API_SESSION="your_waha_session_name" # Defaults to 'default'
```

Replace placeholders with your actual credentials.

# Usage

1. Start Application:

With virtual environment active and configurations set:

```env
streamlit run main.py
```

2. Access UI:

Open the URL provided by Streamlit, usually http://localhost:8501

3. Configure in Sidebar:

* Enter Google and Exa API Keys.
* (Optional) Provide LinkAce and Waha details.
* Select a Gemini model.

4. Enter Topic & Generate:

Input the newsletter topic and click "Generate Newsletter".

5. View, Download, & Publish:

The final newsletter is displayed for viewing/download (Markdown). If configured, it's sent to the specified WhatsApp chat, with publishing status shown.

6. Advanced Options:

Optionally, "Show all intermediate agent results" to view pipeline outputs.

# Agent Pipeline Stages

1. Stage 0: Ingestion (LinkAce) (Optional)

* Agent: `ingest_agent` with `LinkAceSearchTool`.
* Action: Fetches recent links from the configured LinkAce list.

2. Stage 0.5: Initial Web Search (Exa)

* Action: Direct web search via `WebSearchTool` (Exa API) on the input topic.

3. Stage 1: Researching

* Agent: `researcher_agent` with `WebSearchTool` (Exa API).
* Action: Synthesizes LinkAce data and Exa search results; performs further Exa searches for comprehensive understanding.

4. Stage 2: Generating Insights

* Agent: `insights_expert_agent` with `WebSearchTool` (Exa API).
* Action: Analyzes research, using Exa for verification/expansion, providing deeper analysis and context.

5. Stage 3: Drafting Newsletter

* Agent: `writer_agent`.
* Action: Transforms insights into an engaging, accessible newsletter draft.

6. Stage 4: Editing Newsletter

* Agent: `editor_agent`.
* Action: Proofreads, refines, and structures the draft for clarity and publication readiness.

7. Stage 5: Publishing to WhatsApp (Optional)

* Agent: `publish_agent` with `WahaPublishTool`.
* Action: Sends the final newsletter to the specified WhatsApp chat, using "seen" and "typing" indicators.

# Contributing

Contributions are welcome! For major changes, please open an issue first. For minor fixes, submit a pull request.
(Standard fork, branch, commit, push, PR process.)

# License

Distributed under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.
