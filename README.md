# Agentic Research Assistant 
The Agentic Research Assistant is built as a multi‐stage pipeline that turns your free‐form query into a targeted research session. Rather than presenting raw search results, it drives a sequence of classification, retrieval, synthesis, and suggestion steps—each implemented as a modular “tool” that can be independently tested, replaced, or extended. Under the hood, it uses vector embeddings, a remote index service, and an LLM orchestrator to keep everything decoupled yet smoothly integrated.

## Live Demo
Try out the Agentic Research Assistant live on Streamlit:
[👉 Launch the live demo](https://m4d7kedkaqbqiqnevzp7kj.streamlit.app/)

# Session Dashboard

Totals & Duration: Tracks your message count, questions sent, agent responses, and elapsed time to gauge session activity.

# Search Controls / Display Toggles 

- Sources Slider: Sets how many top‐K documents the agent retrieves (1–10).
- Creativity Slider: Adjusts the LLM’s temperature for more “by-the-book” vs. more exploratory summaries.
- Timestamps & Auto-Scroll: Show message times and keep the latest reply in view.

# Agentic Workflow

- Dynamic Routing: LLM classifies each query as academic, industry, or both, then hits the appropriate Pinecone index.

- Transparent Retrieval: Retrieved snippets feed into a summary that always ends with a numbered Sources & References list (title, score, excerpt).

- Proactive Suggestions: After each answer, you get 2–3 follow-up questions and 2–3 research-direction prompts to keep exploring.
