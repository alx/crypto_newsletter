import patch # type: ignore

import streamlit as st
import os
import time
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
import random

from karo.core.base_agent import BaseAgent, BaseAgentConfig
from karo.providers.gemini_provider import GeminiProvider, GeminiProviderConfig
from karo.prompts.system_prompt_builder import SystemPromptBuilder
from karo.tools.base_tool import BaseTool
from karo.schemas.base_schemas import BaseInputSchema

from pydantic import BaseModel, Field

from web_search_tool import WebSearchTool, WebSearchInputSchema # Assuming this remains for Exa

load_dotenv()

# --- LinkAce Search Tool Definition ---
class LinkAceSearchInputSchema(BaseModel):
    """Schema for LinkAce search tool inputs"""
    list_id: str = Field(..., description="The ID of the LinkAce list to search")
    days_ago: int = Field(default=1, description="How many days back to search for content (e.g., 1 for last 24 hours)")

class LinkAceSearchResultItem(BaseModel):
    """Schema for a single link item from LinkAce"""
    url: str
    title: Optional[str] = None
    description: Optional[str] = None

class LinkAceSearchOutputSchema(BaseModel):
    """Schema for LinkAce search tool outputs"""
    success: bool = Field(..., description="Whether the LinkAce API call was successful")
    links: Optional[List[LinkAceSearchResultItem]] = Field(None, description="List of links found")
    error_message: Optional[str] = Field(None, description="Error message if the call failed")
    list_id_searched: Optional[str] = Field(None, description="The list ID that was searched")

class LinkAceSearchTool(BaseTool):
    """
    Tool for fetching links from a specific LinkAce list, filtered by creation date.
    """
    name: str = "linkace_search_tool"
    description: str = (
        "Fetches links from a specified LinkAce list that were created within a given number of past days. "
        "Requires LinkAce instance URL, API token, and the list ID."
    )
    input_schema = LinkAceSearchInputSchema
    output_schema = LinkAceSearchOutputSchema

    def __init__(self, linkace_url: str, linkace_token: str):
        super().__init__()
        if not linkace_url or not linkace_token:
            raise ValueError("LinkAce URL and Token are required for LinkAceSearchTool.")
        self.linkace_url = linkace_url.rstrip('/')
        self.linkace_token = linkace_token

    def run(self, input_data: LinkAceSearchInputSchema) -> Dict[str, Any]:
        try:
            from_datetime_str = (datetime.now(timezone.utc) - timedelta(days=input_data.days_ago)).strftime("%Y-%m-%d")
            api_url = f"{self.linkace_url}/api/v1/links"
            headers = {
                "Authorization": f"Bearer {self.linkace_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            params = {
                "list_id": input_data.list_id,
                "created_at_from": from_datetime_str,
                "limit": 100,
                "sort_by": "created_at",
                "sort_order": "desc",
            }
            # st.write(f"Querying LinkAce: {api_url} with params: {params}") # Debug
            response = requests.get(api_url, headers=headers, params=params, timeout=20)
            response.raise_for_status()
            response_data = response.json()
            fetched_links: List[LinkAceSearchResultItem] = []
            for item in response_data.get("data", []):
                description = item.get("description")
                if description and len(description) > 300:
                    description = description[:297] + "..."
                fetched_links.append(
                    LinkAceSearchResultItem(
                        url=item.get("url", "URL not found"),
                        title=item.get("title"),
                        description=description,
                    )
                )
            # st.write(f"LinkAce tool found {len(fetched_links)} links.") # Debug
            return {
                "success": True,
                "links": [link.model_dump() for link in fetched_links],
                "error_message": None,
                "list_id_searched": input_data.list_id
            }
        except requests.exceptions.HTTPError as e:
            error_msg = f"LinkAce API HTTP error: {e.response.status_code} - {e.response.text}"
            # st.error(f"LinkAce Tool Error: {error_msg}") # Debug
            return {"success": False, "links": None, "error_message": error_msg, "list_id_searched": input_data.list_id}
        except requests.exceptions.RequestException as e:
            error_msg = f"LinkAce API request failed: {str(e)}"
            # st.error(f"LinkAce Tool Error: {error_msg}") # Debug
            return {"success": False, "links": None, "error_message": error_msg, "list_id_searched": input_data.list_id}
        except Exception as e:
            error_msg = f"Error processing LinkAce links: {str(e)}"
            # st.error(f"LinkAce Tool Error: {error_msg}") # Debug
            return {"success": False, "links": None, "error_message": error_msg, "list_id_searched": input_data.list_id}

# --- Waha WhatsApp Publish Tool Definition ---
class WahaPublishInputSchema(BaseModel):
    """Schema for Waha WhatsApp publishing tool inputs"""
    chat_id: str = Field(..., description="Target WhatsApp Chat ID (e.g., 1234567890@c.us or group_id@g.us)")
    text: str = Field(..., description="The text message to send")

class WahaPublishOutputSchema(BaseModel):
    """Schema for Waha WhatsApp publishing tool outputs"""
    success: bool = Field(..., description="Whether the message was sent successfully")
    message_id: Optional[str] = Field(None, description="The ID of the sent message if successful")
    status_message: str = Field(..., description="A message describing the outcome of the publishing attempt")

class WahaPublishTool(BaseTool):
    """
    Tool for publishing text messages to WhatsApp using the Waha (WhatsApp HTTP API).
    Implements anti-blocking measures like sending seen, typing indicators, and delays.
    """
    name: str = "whatsapp_publish_tool"
    description: str = (
        "Sends a text message to a specified WhatsApp chat ID via Waha. "
        "Includes measures to reduce the risk of blocking."
    )
    input_schema = WahaPublishInputSchema
    output_schema = WahaPublishOutputSchema

    def __init__(self, waha_api_host: str, waha_session: str):
        super().__init__()
        if not waha_api_host or not waha_session:
            raise ValueError("Waha API Host and Session are required for WahaPublishTool.")
        self.waha_api_host = waha_api_host.rstrip('/')
        self.waha_session = waha_session
        self.headers = {"Content-Type": "application/json"}

    def _call_waha_api(self, endpoint: str, payload: Optional[Dict[str, Any]] = None, method: str = "POST") -> requests.Response:
        """Helper function to call Waha API endpoints."""
        url = f"{self.waha_api_host}/api/sessions/{self.waha_session}/{endpoint}"
        if method.upper() == "POST":
            return requests.post(url, json=payload, headers=self.headers, timeout=20)
        elif method.upper() == "GET": # If any GET endpoints are needed later
            return requests.get(url, params=payload, headers=self.headers, timeout=20)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

    def run(self, input_data: WahaPublishInputSchema) -> Dict[str, Any]:
        chat_id = input_data.chat_id
        text_to_send = input_data.text

        try:
            # 1. Send Seen
            st.info(f"[WAHA] Sending 'seen' to {chat_id}...")
            seen_payload = {"chatId": chat_id}
            # Note: /sendSeen is often a general endpoint. If it's chat-specific, the path might be different.
            # Using /chats/{chatId}/ack might be for a specific messageId.
            # The user specifically mentioned POST /api/sendSeen/.
            # Waha docs usually show /api/sessions/{session}/sendSeen with {"chatId": "..."}
            try:
                # Using the more specific endpoint if available, or a general one.
                # For now, assuming `sendSeen` is a valid endpoint for the session.
                response_seen = self._call_waha_api("sendSeen", payload=seen_payload)
                if response_seen.status_code == 200:
                    st.success(f"[WAHA] 'Seen' sent to {chat_id}.")
                else:
                    st.warning(f"[WAHA] Could not send 'seen' to {chat_id}. Status: {response_seen.status_code}, Response: {response_seen.text[:200]}")
            except Exception as e_seen:
                st.warning(f"[WAHA] Error sending 'seen' to {chat_id}: {str(e_seen)}")


            # 2. Start Typing
            st.info(f"[WAHA] Starting 'typing' for {chat_id}...")
            typing_payload = {"chatId": chat_id}
            response_typing_start = self._call_waha_api("startTyping", payload=typing_payload)
            if response_typing_start.status_code != 200:
                st.warning(f"[WAHA] Could not start 'typing' for {chat_id}. Status: {response_typing_start.status_code}")

            # 3. Wait for a random interval
            # Average reading speed: ~200 words per minute. Words are ~5 chars. So 1000 chars/min.
            # ~16 chars/sec. Delay = len(text) / 16. Add some randomness.
            delay = random.uniform(1.5, 3.0) + (len(text_to_send) / 100.0) # 1 sec per 100 chars
            delay = min(delay, 10) # Cap delay at 10 seconds
            st.info(f"[WAHA] Waiting for {delay:.2f} seconds before sending message...")
            time.sleep(delay)

            # 4. Send Text Message
            st.info(f"[WAHA] Sending message to {chat_id}...")
            send_payload = {"chatId": chat_id, "text": text_to_send}
            response_send = self._call_waha_api("sendText", payload=send_payload)

            # 5. Stop Typing (optional, often cleared by sending message)
            try:
                st.info(f"[WAHA] Stopping 'typing' for {chat_id}...")
                response_typing_stop = self._call_waha_api("stopTyping", payload=typing_payload)
                if response_typing_stop.status_code != 200:
                    st.warning(f"[WAHA] Could not stop 'typing' for {chat_id}. Status: {response_typing_stop.status_code}")
            except Exception as e_stop_typing:
                 st.warning(f"[WAHA] Error stopping 'typing' for {chat_id}: {str(e_stop_typing)}")


            if response_send.status_code == 200 or response_send.status_code == 201:
                message_id = response_send.json().get("id")
                st.success(f"[WAHA] Message sent successfully to {chat_id}. Message ID: {message_id}")
                return {
                    "success": True,
                    "message_id": message_id,
                    "status_message": f"Message sent successfully to {chat_id}. Message ID: {message_id}"
                }
            else:
                error_details = response_send.text[:500] # Limit error message length
                st.error(f"[WAHA] Failed to send message to {chat_id}. Status: {response_send.status_code}, Response: {error_details}")
                return {
                    "success": False,
                    "status_message": f"Failed to send message. Status: {response_send.status_code}, Error: {error_details}"
                }

        except requests.exceptions.RequestException as e:
            st.error(f"[WAHA] API request failed: {str(e)}")
            return {"success": False, "status_message": f"API request failed: {str(e)}"}
        except Exception as e:
            st.error(f"[WAHA] Error publishing to WhatsApp: {str(e)}")
            return {"success": False, "status_message": f"Error publishing to WhatsApp: {str(e)}"}


class NewsletterAgents:
    def __init__(self,
                 model_name: str = "gemini-1.5-flash-latest",
                 google_api_key: str = None,
                 exa_api_key: str = None,
                 linkace_url: Optional[str] = None,
                 linkace_token: Optional[str] = None,
                 linkace_list_id: Optional[str] = None,
                 waha_api_host: Optional[str] = None,
                 waha_session: Optional[str] = None):

        self.model_name = model_name
        self.google_api_key = google_api_key
        self.exa_api_key = exa_api_key or os.getenv("EXA_API_KEY")

        self.linkace_url = linkace_url or os.getenv("LINKACE_URL")
        self.linkace_token = linkace_token or os.getenv("LINKACE_TOKEN")
        self.linkace_list_id = linkace_list_id or os.getenv("LINKACE_LIST_ID")

        self.waha_api_host = waha_api_host or os.getenv("WAHA_API_HOST", "http://swyn:3000/")
        self.waha_session = waha_session or os.getenv("WAHA_API_SESSION", "default")

        self.web_search_tool = WebSearchTool(api_key=self.exa_api_key)

        self.linkace_search_tool: Optional[LinkAceSearchTool] = None
        if self.linkace_url and self.linkace_token:
            try:
                self.linkace_search_tool = LinkAceSearchTool(linkace_url=self.linkace_url, linkace_token=self.linkace_token)
            except ValueError as e:
                st.warning(f"Could not initialize LinkAceSearchTool: {e}")

        self.waha_publish_tool: Optional[WahaPublishTool] = None
        if self.waha_api_host and self.waha_session:
            try:
                self.waha_publish_tool = WahaPublishTool(waha_api_host=self.waha_api_host, waha_session=self.waha_session)
            except ValueError as e:
                st.warning(f"Could not initialize WahaPublishTool: {e}")

        self.ingest_agent = self._create_ingest_agent()
        self.researcher = self._create_researcher_agent()
        self.insights_expert = self._create_insights_expert_agent()
        self.writer = self._create_writer_agent()
        self.editor = self._create_editor_agent()
        self.publish_agent = self._create_publish_agent()


    def _create_gemini_provider_config(self, temperature: float = 0.1) -> GeminiProviderConfig:
        return GeminiProviderConfig(
            model=self.model_name,
            api_key=self.google_api_key,
            temperature=temperature,
        )

    def _create_ingest_agent(self) -> BaseAgent:
        provider_config = self._create_gemini_provider_config(temperature=0.0)
        provider = GeminiProvider(config=provider_config)
        available_tools = [self.linkace_search_tool] if self.linkace_search_tool else []
        prompt_builder = SystemPromptBuilder(
            role_description="You are an Ingestion Specialist.",
            core_instructions=[
                "Use `linkace_search_tool` if available to fetch links from the provided LinkAce list ID for the last day.",
                "If the tool is not available or fails, state that links could not be fetched."
            ],
            output_instructions=[
                "If links are fetched, list each link: Title, URL, Description.",
                "If no links, state: 'No new links found in LinkAce for list ID [list_id] in the last day.'",
                "If tool fails, report the error."
            ]
        )
        agent_config = BaseAgentConfig(
            provider_config=provider_config, prompt_builder=prompt_builder, tools=available_tools,
            max_tool_call_attempts=2, tool_sys_msg="Use `linkace_search_tool` if available and list ID is given."
        )
        return BaseAgent(config=agent_config)

    def _create_researcher_agent(self) -> BaseAgent:
        provider_config = self._create_gemini_provider_config(temperature=0.1)
        provider = GeminiProvider(config=provider_config)
        available_tools = [self.web_search_tool]
        prompt_builder = SystemPromptBuilder(
            role_description="You are an AI Researcher.",
            core_instructions=[
                "Synthesize information from LinkAce links (if provided) AND general web searches via `web_search_tool`.",
                "MUST call `web_search_tool` for supplementary research.",
                "Provide comprehensive research with sources, integrating all findings."
            ],
            output_instructions=[
                "1. Analyze LinkAce links.", "2. Call `web_search_tool`.",
                "3. Organize combined findings with source links.", "4. Highlight impact."
            ]
        )
        agent_config = BaseAgentConfig(
            provider_config=provider_config, prompt_builder=prompt_builder, tools=available_tools,
            max_tool_call_attempts=5, tool_sys_msg="MUST use `web_search_tool` for additional info."
        )
        return BaseAgent(config=agent_config)

    def _create_insights_expert_agent(self) -> BaseAgent:
        provider_config = self._create_gemini_provider_config(temperature=0.1)
        provider = GeminiProvider(config=provider_config)
        available_tools = [self.web_search_tool]
        prompt_builder = SystemPromptBuilder(
            role_description="You are an AI Insights Expert.",
            core_instructions=[
                "PRIMARY task: use `web_search_tool` to verify, expand upon, and contextualize provided research.",
                "MUST call `web_search_tool`.",
                "Provide detailed analysis on significance, applications, and future potential."
            ],
            output_instructions=[
                "1. Call `web_search_tool`.", "2. Organize analysis.",
                "3. Include industry implications and future directions."
            ]
        )
        agent_config = BaseAgentConfig(
           provider_config=provider_config, prompt_builder=prompt_builder, tools=available_tools,
            max_tool_call_attempts=5, tool_sys_msg="MUST use `web_search_tool` for your analysis."
        )
        return BaseAgent(config=agent_config)

    def _create_writer_agent(self) -> BaseAgent:
        provider_config = self._create_gemini_provider_config(temperature=0.7)
        provider = GeminiProvider(config=provider_config)
        prompt_builder = SystemPromptBuilder(
            role_description="You are a Newsletter Content Creator.",
            core_instructions=[
                "Transform insights into engaging, reader-friendly newsletter content.",
                "Make complex topics accessible. Highlight innovation, relevance, and impact."
            ],
            output_instructions="Professional yet engaging tone. Clear headings, concise paragraphs."
        )
        agent_config = BaseAgentConfig(provider_config=provider_config, prompt_builder=prompt_builder)
        return BaseAgent(config=agent_config)

    def _create_editor_agent(self) -> BaseAgent:
        provider_config = self._create_gemini_provider_config(temperature=0.2)
        provider = GeminiProvider(config=provider_config)
        prompt_builder = SystemPromptBuilder(
            role_description="You are a meticulous Newsletter Editor.",
            core_instructions=[
                "Proofread, refine, and structure the newsletter for publication.",
                "Ensure clarity, eliminate errors, enhance readability, align tone."
            ],
            output_instructions="Include valid URLs. Format with headings, bullets. Explain technical terms."
        )
        agent_config = BaseAgentConfig(provider_config=provider_config, prompt_builder=prompt_builder)
        return BaseAgent(config=agent_config)

    def _create_publish_agent(self) -> BaseAgent:
        provider_config = self._create_gemini_provider_config(temperature=0.0) # Low temp for direct action
        provider = GeminiProvider(config=provider_config)
        available_tools = [self.waha_publish_tool] if self.waha_publish_tool else []

        prompt_builder = SystemPromptBuilder(
            role_description="You are a WhatsApp Publishing Specialist.",
            core_instructions=[
                "Your task is to send a given text message to a specified WhatsApp chat ID using the `whatsapp_publish_tool`.",
                "The user will provide the final newsletter text and the target WhatsApp chat ID.",
                "You MUST use the `whatsapp_publish_tool` to send the message if the tool is available.",
                "If the tool is not available, state that publishing is not possible."
            ],
            output_instructions=[
                "If the message is sent successfully by the tool, confirm this and include any message ID returned by the tool.",
                "If the tool reports an error or fails to send the message, report the error message provided by the tool.",
                "Example success: 'Message successfully sent to [chat_id] via WhatsApp. Message ID: [message_id].'",
                "Example failure: 'Failed to send message to [chat_id] via WhatsApp. Error: [tool_error_message].'"
            ]
        )

        agent_config = BaseAgentConfig(
            provider_config=provider_config,
            prompt_builder=prompt_builder,
            tools=available_tools,
            max_tool_call_attempts=1, # Only one attempt to publish
            tool_sys_msg="You MUST use the `whatsapp_publish_tool` to send the provided text to the given chat ID."
        )
        return BaseAgent(config=agent_config)

    def manual_search(self, query: str, days_ago: int = 7) -> dict:
        if not self.exa_api_key:
            st.warning("Exa API key not configured. Manual search skipped.")
            return {"success": False, "error_message": "Exa API key not configured.", "results": []}
        st.info(f"Searching (Exa): '{query}'...")
        search_input = WebSearchInputSchema(search_query=query, days_ago=days_ago, max_results=5)
        try:
            results = self.web_search_tool.run(search_input)
            if results.get("success"):
                st.success(f"Exa Search successful: {results.get('total_results_found', 0)} results for '{query}'")
                return results
            st.error(f"Exa Search failed for '{query}': {results.get('error_message', 'Unknown error')}")
            return {"success": False, "error_message": results.get('error_message', 'Unknown error'), "results": []}
        except Exception as e:
            st.error(f"Exa Search error for '{query}': {str(e)}")
            return {"success": False, "error_message": str(e), "results": []}

    def _get_agent_response_content(self, result: Any) -> str:
        if hasattr(result, 'response_message') and result.response_message: return result.response_message
        if hasattr(result, 'content') and result.content: return result.content
        return str(result)

    def run_pipeline(self, user_input_topic: str, target_whatsapp_chat_id: Optional[str] = None) -> Dict[str, Any]:
        # Stage 0: Ingest links from LinkAce
        ingested_content_summary = "LinkAce not configured or ingestion skipped.\n"
        if self.linkace_search_tool and self.linkace_list_id and self.ingest_agent.config.tools:
            with st.status(f"Stage 0: Ingesting from LinkAce list ID: {self.linkace_list_id}..."):
                ingest_input = BaseInputSchema(chat_message=f"Fetch links from LinkAce list ID '{self.linkace_list_id}'.")
                try:
                    ingest_result = self.ingest_agent.run(ingest_input)
                    ingested_content_summary = self._get_agent_response_content(ingest_result)
                    st.success(f"LinkAce ingestion complete.")
                except Exception as e:
                    st.error(f"Error running ingest agent: {str(e)}")
                    ingested_content_summary = f"LinkAce ingestion agent failed: {str(e)}\n"
        st.markdown("### Ingested Links (from LinkAce if configured)")
        st.text_area("Ingested Content Summary", ingested_content_summary, height=100, key="ingest_sum")

        # Stage 0.5: Initial Web Search (Exa)
        search_summary_for_researcher = "No general web search results.\n"
        with st.status(f"Searching web for '{user_input_topic}'..."):
            primary_search_results = self.manual_search(f"latest developments in {user_input_topic}", days_ago=7)
        temp_search_summary = "WEB SEARCH RESULTS (Exa):\n"
        if primary_search_results.get("success") and primary_search_results.get("results"):
            for i, res in enumerate(primary_search_results.get("results", [])):
                temp_search_summary += f"[Res {i+1}] Title: {res.get('title')}\nURL: {res.get('url')}\nPreview: {res.get('content_preview', '')[:100]}...\n\n"
            search_summary_for_researcher = temp_search_summary
        else: temp_search_summary += "No results or search failed.\n"; search_summary_for_researcher = temp_search_summary
        st.markdown("### Web Search Results (Exa)")
        st.text_area("Exa Search Summary", search_summary_for_researcher, height=100, key="exa_sum")

        # --- Agent Pipeline ---
        stages = [
            ("Stage 1: Researching", self.researcher,
             lambda: (f"Research Task: Synthesize info about '{user_input_topic}'.\n\n"
                      f"LINKACE CONTEXT:\n{ingested_content_summary}\n\n"
                      f"EXA WEB SEARCH CONTEXT:\n{search_summary_for_researcher}\n\n"
                      f"Instructions: Analyze context, then use `web_search_tool` for more. Cite sources.")),
            ("Stage 2: Generating Insights", self.insights_expert,
             lambda: (f"Analyze this research on '{user_input_topic}' for deep insights:\n\n{pipeline_results['research']}\n\n"
                      f"Use `web_search_tool` if needed for verification or expansion.")),
            ("Stage 3: Drafting Newsletter", self.writer,
             lambda: f"Transform these insights on '{user_input_topic}' into a newsletter:\n\n{pipeline_results['insights']}"),
            ("Stage 4: Editing Newsletter", self.editor,
             lambda: f"Proofread and refine this draft on '{user_input_topic}':\n\n{pipeline_results['draft']}")
        ]

        pipeline_results: Dict[str, Any] = {
            "ingested_links": ingested_content_summary,
            "exa_searches": search_summary_for_researcher,
        }
        current_content = ""

        for stage_name, agent, input_generator in stages:
            with st.status(f"{stage_name}..."):
                agent_input_msg = input_generator()
                agent_input_obj = BaseInputSchema(chat_message=agent_input_msg)
                try:
                    agent_result_obj = agent.run(agent_input_obj, history=[])
                    current_content = self._get_agent_response_content(agent_result_obj)
                    st.success(f"{stage_name} complete.")
                except Exception as e:
                    st.error(f"Error in {stage_name}: {str(e)}")
                    current_content = f"{stage_name} failed: {str(e)}"

                # Store result based on stage name (lowercase, replace space with underscore)
                result_key = stage_name.split(":")[1].strip().lower().replace(" ", "_").replace("newsletter", "").strip("_")
                if "researching" in stage_name.lower(): result_key = "research"
                elif "insights" in stage_name.lower(): result_key = "insights"
                elif "drafting" in stage_name.lower(): result_key = "draft"
                elif "editing" in stage_name.lower(): result_key = "final" # Editor produces final

                pipeline_results[result_key] = current_content

        # Stage 5: Publishing to WhatsApp
        publish_status_message = "WhatsApp publishing skipped (no chat ID provided or tool not configured)."
        if target_whatsapp_chat_id and self.waha_publish_tool and self.publish_agent.config.tools:
            with st.status(f"Stage 5: Publishing to WhatsApp ({target_whatsapp_chat_id})..."):
                final_newsletter_text = pipeline_results.get("final", "Error: Final newsletter content not available.")
                if "failed" in final_newsletter_text.lower() or "not available" in final_newsletter_text.lower() :
                     st.error("Cannot publish as final newsletter generation failed or is unavailable.")
                     publish_status_message = "Publishing skipped: Final newsletter content unavailable."
                else:
                    publish_input_msg = (
                        f"Publish the following newsletter text to WhatsApp chat ID '{target_whatsapp_chat_id}'.\n\n"
                        f"Newsletter Text:\n{final_newsletter_text}"
                    )
                    publish_input = BaseInputSchema(chat_message=publish_input_msg)
                    try:
                        publish_result_obj = self.publish_agent.run(publish_input)
                        # The publish_agent's output_instructions should ensure it returns the tool's direct status.
                        # So, the agent's response itself is the status message.
                        publish_status_message = self._get_agent_response_content(publish_result_obj)
                        if "success" in publish_status_message.lower():
                            st.success(f"WhatsApp publishing attempt finished. Agent says: {publish_status_message}")
                        else:
                            st.warning(f"WhatsApp publishing attempt finished. Agent says: {publish_status_message}")
                    except Exception as e:
                        st.error(f"Error running publish agent: {str(e)}")
                        publish_status_message = f"Publishing agent failed: {str(e)}"
        pipeline_results["publish_status"] = publish_status_message

        return pipeline_results

def main():
    st.set_page_config(page_title="AI Newsletter Generator (Gemini + Waha)", page_icon="📰", layout="wide")
    st.title("AI Newsletter Generator")
    st.subheader("Gemini & Exa Search, with LinkAce Ingestion & WhatsApp Publishing")

    st.sidebar.header("API Configuration")
    google_api_key = st.sidebar.text_input("Google API Key (Gemini)", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
    exa_api_key = st.sidebar.text_input("Exa API Key", value=os.getenv("EXA_API_KEY", ""), type="password")

    st.sidebar.header("LinkAce Configuration (Optional)")
    linkace_url = st.sidebar.text_input("LinkAce URL", value=os.getenv("LINKACE_URL", ""))
    linkace_token = st.sidebar.text_input("LinkAce API Token", value=os.getenv("LINKACE_TOKEN", ""), type="password")
    linkace_list_id = st.sidebar.text_input("LinkAce List ID", value=os.getenv("LINKACE_LIST_ID", ""))

    st.sidebar.header("WhatsApp Publishing (Optional)")
    waha_target_chat_id = st.sidebar.text_input("Target WhatsApp Chat ID", placeholder="e.g., 1234567890@c.us")
    waha_api_host_ui = st.sidebar.text_input("Waha API Host", value=os.getenv("WAHA_API_HOST", "http://swyn:3000/"))
    waha_session_ui = st.sidebar.text_input("Waha API Session", value=os.getenv("WAHA_API_SESSION", "default"))

    st.sidebar.header("Model Configuration")
    model_name = st.sidebar.selectbox("Select Gemini Model", ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-pro"], index=0)

    if google_api_key: os.environ["GOOGLE_API_KEY"] = google_api_key
    if exa_api_key: os.environ["EXA_API_KEY"] = exa_api_key
    # LinkAce and Waha details are passed to constructor, no need to set env here if taken from UI.

    if not google_api_key: st.warning("Please enter Google API key for Gemini.")
    if not exa_api_key: st.warning("Please enter Exa API key for web search.")

    st.subheader("Newsletter Topic")
    topic = st.text_input("Enter topic:", placeholder="e.g., AI in healthcare")

    with st.expander("Advanced Options"):
        show_intermediate = st.checkbox("Show all intermediate agent results", value=False)

    can_generate = bool(topic and google_api_key and exa_api_key)
    generate_btn = st.button("Generate Newsletter", type="primary", disabled=not can_generate)

    if generate_btn:
        if not can_generate:
            st.error("Provide topic, Google & Exa API keys.")
        else:
            try:
                agents = NewsletterAgents(
                    model_name=model_name, google_api_key=google_api_key, exa_api_key=exa_api_key,
                    linkace_url=linkace_url, linkace_token=linkace_token, linkace_list_id=linkace_list_id,
                    waha_api_host=waha_api_host_ui, waha_session=waha_session_ui
                )
                start_time = time.time()
                with st.spinner(f"Generating newsletter about '{topic}'..."):
                    result = agents.run_pipeline(topic, target_whatsapp_chat_id=waha_target_chat_id)
                st.success(f"Newsletter pipeline finished in {time.time() - start_time:.2f}s!")

                st.subheader("Your AI Newsletter")
                st.markdown(result.get("final", "Not generated."))
                st.download_button("Download Newsletter (MD)", result.get("final", ""), f"ai_newsletter_{topic.replace(' ', '_')}.md", "text/markdown")

                if result.get("publish_status") and "skipped" not in result["publish_status"].lower() :
                    st.subheader("WhatsApp Publishing Status")
                    if "success" in result["publish_status"].lower() or "sent successfully" in result["publish_status"].lower():
                        st.success(result["publish_status"])
                    else:
                        st.warning(result["publish_status"])

                if show_intermediate:
                    st.markdown("---"); st.subheader("Intermediate Results")
                    for key, val in result.items():
                        if key not in ["final", "publish_status"]:
                             with st.expander(f"{key.replace('_', ' ').title()}"): st.markdown(str(val))
            except Exception as e:
                st.error(f"Error generating newsletter: {str(e)}"); st.exception(e)

    st.sidebar.markdown("---"); st.sidebar.subheader("About")
    st.sidebar.info("Multi-stage AI newsletter generator using Gemini, Exa, LinkAce, and Waha (WhatsApp).")

if __name__ == "__main__":
    main()
