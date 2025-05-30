import patch

import streamlit as st
import os
import time
from dotenv import load_dotenv


from karo.core.base_agent import BaseAgent, BaseAgentConfig
from karo.providers.openai_provider import OpenAIProvider, OpenAIProviderConfig
from karo.prompts.system_prompt_builder import SystemPromptBuilder
from typing import Dict, Any


from web_search_tool import WebSearchTool, WebSearchInputSchema


load_dotenv()

class NewsletterAgents:
    def __init__(self, model_name: str = "gpt-4-turbo", api_key: str = None, exa_api_key: str = None):

        self.model_name = model_name
        self.api_key = api_key
        self.exa_api_key = exa_api_key or os.getenv("EXA_API_KEY")
        

        self.web_search_tool = WebSearchTool(api_key=self.exa_api_key)
        

        self.researcher = self._create_researcher_agent()
        self.insights_expert = self._create_insights_expert_agent()
        self.writer = self._create_writer_agent()
        self.editor = self._create_editor_agent()
    
    def _create_researcher_agent(self) -> BaseAgent:

        provider_config = OpenAIProviderConfig(
            model=self.model_name, 
            api_key=self.api_key,
            tool_choice="auto",
            temperature=0.1,  
        )
        provider = OpenAIProvider(config=provider_config)
        

        available_tools = [self.web_search_tool]
        
 
        prompt_builder = SystemPromptBuilder(
            role_description="You are an AI Researcher tracking the latest advancements and trends in AI, machine learning, and deep learning.",
            core_instructions=(
                "Your PRIMARY task is to use the web_search_tool tool to find the latest information. "
                "THIS IS CRITICAL: You MUST make at least one call to the web_search_tool tool - it is your main job. "
                "Without using this tool, your response will be incomplete and outdated. "
                "After using the search tool, provide comprehensive research with reliable sources. "
                "Include exact search queries you used and summarize the most relevant findings."
            ),
            output_instructions=(
                "1. FIRST: Call the web_search_tool tool with an appropriate query.\n"
                "2. THEN: Organize your findings into clear sections with source links.\n"
                "3. ALWAYS: Highlight the potential impact of each development."
            )
        )
        
 
        agent_config = BaseAgentConfig(
            provider_config=provider_config,
            prompt_builder=prompt_builder,
            tools=available_tools,
            max_tool_call_attempts=5,  
            tool_sys_msg="You have access to the web_search_tool tool. You MUST use this tool to find information."
        )
        

        return BaseAgent(config=agent_config)
    
    def _create_insights_expert_agent(self) -> BaseAgent:
 
        provider_config = OpenAIProviderConfig(
            model=self.model_name, 
            api_key=self.api_key,
            tool_choice="auto",
            temperature=0.1, 
        )
        provider = OpenAIProvider(config=provider_config)
        

        available_tools = [self.web_search_tool]
        
        prompt_builder = SystemPromptBuilder(
            role_description="You are an AI Insights Expert with deep knowledge of the field of AI.",
            core_instructions=(
                "Your PRIMARY task is to use the web_search_tool tool to verify and expand upon the research provided. "
                "THIS IS CRITICAL: You MUST make at least one call to the web_search_tool tool - it is your main job. "
                "Without using this tool, your insights will be incomplete. "
                "After searching, provide detailed analysis on the significance, applications, and future potential of each development."
            ),
            output_instructions=(
                "1. FIRST: Call the web_search_tool tool to verify and expand upon the research.\n"
                "2. THEN: Organize your analysis into clear sections.\n"
                "3. ALWAYS: Include potential industry implications and future directions."
            )
        )

        agent_config = BaseAgentConfig(
           provider_config=provider_config,
            prompt_builder=prompt_builder,
            tools=available_tools,
            max_tool_call_attempts=5,
            tool_sys_msg="You have access to the search_and_contents tool. You MUST use this tool to find information."
        )
        
        return BaseAgent(config=agent_config)
    
    def _create_writer_agent(self) -> BaseAgent:

        provider_config = OpenAIProviderConfig(model=self.model_name, api_key=self.api_key)
        provider = OpenAIProvider(config=provider_config)
        

        prompt_builder = SystemPromptBuilder(
            role_description="You are a Newsletter Content Creator with expertise in writing about AI technologies.",
            core_instructions=(
                "Transform insights from the AI Insights Expert into engaging and reader-friendly newsletter content about recent developments in AI, machine learning, and deep learning. "
                "Make complex topics accessible and engaging for a diverse audience. "
                "Transform the insights into reader-friendly content, highlighting the innovation, relevance, and potential impact of each development."
            ),
            output_instructions="Write in a professional yet engaging tone. Structure the content with clear headings and concise paragraphs. Keep the content aligned with the newsletter's goals."
        )

        agent_config = BaseAgentConfig(
            provider_config=provider_config,
            prompt_builder=prompt_builder
        )
        

        return BaseAgent(config=agent_config)
    
    def _create_editor_agent(self) -> BaseAgent:

        provider_config = OpenAIProviderConfig(model=self.model_name, api_key=self.api_key)
        provider = OpenAIProvider(config=provider_config)
        
        prompt_builder = SystemPromptBuilder(
            role_description="You are a meticulous Newsletter Editor for AI content.",
            core_instructions=(
                "Proofread, refine, and structure the newsletter to ensure it is ready for publication. "
                "Maintain professional tone while ensuring content is accessible to the target audience. "
                "Ensure clarity, eliminate errors, enhance readability, and align the tone with the newsletter's vision. "
                "Focus on improving flow, highlighting key insights effectively, and ensuring the newsletter engages the audience."
            ),
            output_instructions="Include valid website URLs to reliable sources for the advancements discussed. Format the newsletter with proper headings, bullet points, and paragraph spacing. Ensure all technical terms are adequately explained for the target audience."
        )

        agent_config = BaseAgentConfig(
            provider_config=provider_config,
            prompt_builder=prompt_builder
        )
        

        return BaseAgent(config=agent_config)
    
    def manual_search(self, query: str, days_ago: int = 7) -> dict:

        st.info(f"Searching for: '{query}'...")
        search_input = WebSearchInputSchema(
            search_query=query,
            days_ago=days_ago,
            max_results=5
        )
        
        try:
            results = self.web_search_tool.run(search_input)
            if results.get("success"):
                st.success(f"Search successful: {results.get('total_results_found', 0)} results found")
                return results
            else:
                st.error(f"Search failed: {results.get('error_message', 'Unknown error')}")
                return {"success": False, "error_message": results.get("error_message", "Unknown error")}
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return {"success": False, "error_message": str(e)}
    
    def run_pipeline(self, user_input: str) -> Dict[str, Any]:

        from karo.schemas.base_schemas import BaseInputSchema

        with st.status("Searching for recent developments..."):
            primary_search_results = self.manual_search(f"latest developments in {user_input}", days_ago=7)
            secondary_search_results = self.manual_search(f"impact of {user_input}", days_ago=14)
        

        search_summary = "SEARCH RESULTS:\n\n"
        if primary_search_results.get("success"):
            search_summary += f"Search for '{primary_search_results.get('search_query')}' found {len(primary_search_results.get('results', []))} results:\n\n"
            for i, result in enumerate(primary_search_results.get("results", [])):
                search_summary += f"[Result {i+1}]\n"
                search_summary += f"Title: {result.get('title')}\n"
                search_summary += f"URL: {result.get('url')}\n"
                search_summary += f"Published: {result.get('published_date')}\n"
                if result.get('content_preview'):
                    search_summary += f"Preview: {result.get('content_preview')[:300]}...\n\n"
        
        if secondary_search_results.get("success"):
            search_summary += f"\nSearch for '{secondary_search_results.get('search_query')}' found {len(secondary_search_results.get('results', []))} results:\n\n"
            for i, result in enumerate(secondary_search_results.get("results", [])):
                search_summary += f"[Result {i+1}]\n"
                search_summary += f"Title: {result.get('title')}\n"
                search_summary += f"URL: {result.get('url')}\n"
                search_summary += f"Published: {result.get('published_date')}\n"
                if result.get('content_preview'):
                    search_summary += f"Preview: {result.get('content_preview')[:300]}...\n\n"

        with st.status("Stage 1: Conducting research..."):
            research_input = BaseInputSchema(
                chat_message=(
                    f"Research task: Analyze these search results about {user_input}.\n\n"
                    f"{search_summary}\n\n"
                    f"Organize these findings into clear research with reliable sources. "
                    f"Include the significance of each development and its broader industry impact. "
                    f"If you need more specific information, use the search_and_contents tool with a specific query."
                )
            )

            research_history = [
                {"role": "user", "content": research_input.chat_message}
            ]
            
            empty_input = BaseInputSchema(chat_message="")
            research_result = self.researcher.run(empty_input, history=research_history)

            if hasattr(research_result, 'response_message'):
                research_content = research_result.response_message
            elif hasattr(research_result, 'content'):
                research_content = research_result.content
            elif hasattr(research_result, 'response_content'):
                research_content = research_result.response_content
            else:
                research_content = str(research_result)
        
        with st.status("Stage 2: Generating insights..."):
            insights_message = (
                f"Add insights to the following research about {user_input}.\n\n"
                f"Research to analyze:\n{research_content}\n\n"
                f"Also consider these additional search results:\n{search_summary[:1000]}...\n\n"
                f"If you need any specific information, use the search_and_contents tool with a specific query."
            )
            
            insights_history = [
                {"role": "user", "content": insights_message}
            ]
            
            insights_result = self.insights_expert.run(empty_input, history=insights_history)

            if hasattr(insights_result, 'response_message'):
                insights_content = insights_result.response_message
            elif hasattr(insights_result, 'content'):
                insights_content = insights_result.content
            elif hasattr(insights_result, 'response_content'):
                insights_content = insights_result.response_content
            else:
                insights_content = str(insights_result)
        
        with st.status("Stage 3: Creating newsletter draft..."):
            writing_message = f"Transform these insights about {user_input} into engaging newsletter content:\n\n{insights_content}"
    
            writing_history = [
                {"role": "user", "content": writing_message}
            ]
            
            writing_result = self.writer.run(empty_input, history=writing_history)
            
            if hasattr(writing_result, 'response_message'):
                newsletter_draft = writing_result.response_message
            elif hasattr(writing_result, 'content'):
                newsletter_draft = writing_result.content
            elif hasattr(writing_result, 'response_content'):
                newsletter_draft = writing_result.response_content
            else:
                newsletter_draft = str(writing_result)
        
        with st.status("Stage 4: Editing and finalizing..."):
            editing_message = (
                f"Proofread and refine this newsletter draft about {user_input}. "
                f"Ensure all sources are properly cited and the content is engaging and informative:\n\n{newsletter_draft}"
            )

            editing_history = [
                {"role": "user", "content": editing_message}
            ]
            
            editing_result = self.editor.run(empty_input, history=editing_history)

            if hasattr(editing_result, 'response_message'):
                final_newsletter = editing_result.response_message
            elif hasattr(editing_result, 'content'):
                final_newsletter = editing_result.content
            elif hasattr(editing_result, 'response_content'):
                final_newsletter = editing_result.response_content
            else:
                final_newsletter = str(editing_result)

        return {
            "research": research_content,
            "insights": insights_content,
            "draft": newsletter_draft,
            "final": final_newsletter
        }

def main():
    st.set_page_config(
        page_title="AI Newsletter Generator",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("AI Newsletter Generator")
    st.subheader("Generate professional newsletters about AI topics in minutes")
    
    st.sidebar.header("Configuration")
    
    openai_api_key = st.sidebar.text_input("OpenAI API Key", 
                                        value=os.getenv("OPENAI_API_KEY", ""), 
                                        type="password",
                                        help="Enter your OpenAI API key")
    
    exa_api_key = st.sidebar.text_input("Exa API Key", 
                                      value=os.getenv("EXA_API_KEY", ""), 
                                      type="password",
                                      help="Enter your Exa API key for web search")
    
    model_name = st.sidebar.selectbox(
        "Select Model", 
        ["gpt-4-turbo", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
        index=0
    )
   
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if exa_api_key:
        os.environ["EXA_API_KEY"] = exa_api_key
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
    if not exa_api_key:
        st.warning("Please enter your Exa API key in the sidebar to enable web search functionality.")
    
    st.subheader("Newsletter Topic")
    topic = st.text_input("Enter a topic for your AI newsletter:", 
                         placeholder="e.g., large language models, computer vision, AI ethics")
    
    with st.expander("Advanced Options"):
        show_intermediate = st.checkbox("Show intermediate results", value=False)
   
    generate_btn = st.button("Generate Newsletter", 
                            type="primary", 
                            disabled=not (topic and openai_api_key and exa_api_key))
    
    if generate_btn and topic:
        try:
            agents = NewsletterAgents(
                model_name=model_name,
                api_key=openai_api_key,
                exa_api_key=exa_api_key
            )
            
            start_time = time.time()
            
            with st.spinner(f"Generating newsletter about '{topic}'... This may take a few minutes."):
                result = agents.run_pipeline(topic)

            time_taken = time.time() - start_time
            st.success(f"Newsletter generated in {time_taken:.2f} seconds!")

            st.subheader("Your AI Newsletter")
            st.markdown(result["final"])
            

            st.download_button(
                label="Download Newsletter (Markdown)",
                data=result["final"],
                file_name=f"ai_newsletter_{topic.replace(' ', '_')}.md",
                mime="text/markdown"
            )

            if show_intermediate:
                with st.expander("Research Results"):
                    st.markdown(result["research"])
                
                with st.expander("Insights Generated"):
                    st.markdown(result["insights"])
                
                with st.expander("Draft Newsletter"):
                    st.markdown(result["draft"])
                
        except Exception as e:
            st.error(f"Error generating newsletter: {str(e)}")
            st.error("Please check your API keys and internet connection and try again.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        """
        This app uses OpenAI models and the Exa search API to generate 
        AI newsletters on any topic. It follows a 4-stage pipeline:
        1. Research
        2. Insights
        3. Writing
        4. Editing
        
        Made with ❤️ using Karo, Streamlit and GPT.
        """
    )

if __name__ == "__main__":
    main()
