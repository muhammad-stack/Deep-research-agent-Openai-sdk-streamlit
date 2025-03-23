import asyncio
import firecrawl
from openai import AsyncOpenAI, api_key
import streamlit as st
from typing import List, Dict, Any
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_default_openai_client, set_tracing_disabled, trace, set_default_openai_key
from firecrawl import FirecrawlApp
from agents.tool import function_tool


# Set page configuration
st.set_page_config(
    page_title="Enhanced Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state for API keys if not exists
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""
if "firecrawl_api_key" not in st.session_state:
    st.session_state.firecrawl_api_key = ""

with st.sidebar:
    st.title("API CONFIGURATION's")

    google_api_key = st.text_input(
        "Google API Key",
        type="password",
        placeholder="Enter your Google API Key",
    )

    firecrawl_api_key = st.text_input(
        "Firecrawl API Key",
        type="password",
        placeholder="Enter your Firecrawl API Key",
    )

    if google_api_key:
        st.session_state.google_api_key = google_api_key

    if firecrawl_api_key:
        st.session_state.firecrawl_api_key = firecrawl_api_key



external_client = AsyncOpenAI(
    api_key=st.session_state.google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

set_default_openai_client(external_client)

set_tracing_disabled(True)

model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=external_client,
)


st.title("Deep Research Assistant !!")
st.markdown("This multi agent system performs enhanced research using "
            "OPENAI agents sdk and Firecrawl.")

research_topic = st.text_input(
    "Enter your research topic:", placeholder="e.g. Latest trends in AI")

@function_tool
async def deep_research(query: str, max_depth: int, time_limit: int, max_urls: int) -> Dict[str, Any]:
    """
    Perform deep research on a given query using Firecrawl and OpenAI.
    """
    try:
        fire_crawl_app = FirecrawlApp(api_key=st.session_state.firecrawl_api_key)
        
        params = {
            "maxDepth": max_depth,
            "timeLimit": time_limit,
            "maxUrls": max_urls,
        }
        
        def on_activity(activity):
            st.write(f"[{activity['type']}] {activity['message']}")
        
        with st.spinner("Performing deep research... This may take a while..."):
            result = fire_crawl_app.deep_research(
                query=query,
                params=params,
                on_activity=on_activity,
            )
            
            # Debug: Print the structure of the result
            st.write("Result structure:", result.keys())
            if 'data' in result:
                st.write("Data keys:", result['data'].keys())
            
            # Use a safer approach to access data
            final_analysis = result.get('data', {}).get('final_analysis', "No final analysis available")
            sources = result.get('data', {}).get('sources', [])
            
            return {
                "success": True,  # Fixed typo in "success"
                "final_analysis": final_analysis,
                "source_count": len(sources),  # Fixed duplicate key
                "sources": sources,  # Renamed key to avoid duplicate
            }
    except Exception as e:
        st.error(f"Deep research error: {str(e)}")
        return {"error": str(e), "success": False}


# Create first agent in our multi-agent system - focused on gathering research
research_agent = Agent(
    name='research_agent',
    instructions="""You are a research assistant that can perform deep web research on any topic.
    When given a research topic or question:
    1. Use the deep_research tool to gather comprehensive information
    - Always use these parameters:
    * max_depth: 3 (for moderate depth)
    * time_limit: 180 (3 minutes)
    * max_urls: 10 (sufficient sources)
    2. The tool will search the web, analyze multiple sources, and provide a synthesis
    3. Review the research results and organize them into a well-structured report
    4. Include proper citations for all sources
    5. Highlight key findings and insights
    """,
    tools=[deep_research],
    model=model
)

# Create second agent in our multi-agent system - focused on elaborative research

elaborative_agent = Agent(
    name='elaborative_agent',
    instructions="""You are an expert content enhancer specializing in research elaboration.
    When given a research report:
    1. Analyze the structure and content of the report
    2. Enhance the report by:
    - Adding more detailed explanations of complex concepts
    - Including relevant examples, case studies, and real-world applications
    - Expanding on key points with additional context and nuance
    - Adding visual elements descriptions (charts, diagrams, infographics)
    - Incorporating latest trends and future predictions
    - Suggesting practical implications for different stakeholders
    3. Maintain academic rigor and factual accuracy
    4. Preserve the original structure while making it more comprehensive
    5. Ensure all additions are relevant and valuable to the topic
    """,
    model=model,
)


async def run_research_process(topic: str):
    """Run the complete multi-agent research process."""
    #  Step1: Inital Research with first agent
    with st.spinner("Agent1 Conducting initial research..."):
        research_result = await Runner.run(research_agent, topic)
        initial_report = research_result.final_output

    with st.expander("Viewing Initial Research Report (Agent 1 Output)"):
        st.markdown(initial_report)

    # Enhance the report with second agent
    with st.spinner("Agent2 Enhancing the research report..."):
        elaboration_input = f""" 
        RESEARCH TOPIC {topic}
        INITIAL RESEARCH REPORT:
        {initial_report}
        Please enhance the report with additional info, examples and case studies
        and deeper insights while maintaining its academic rigor and factual accuracy.
"""

    elaboration_result = await Runner.run(elaborative_agent, elaboration_input)
    enhanced_report = elaboration_result.final_output
    return enhanced_report

if st.button("Start Multi-agent Research", disabled=not (google_api_key and firecrawl_api_key and research_topic)):
    if not google_api_key and firecrawl_api_key:
        st.warning("Please enter your Google or Firecrawl API Key.")
    elif not research_topic:
        st.warning("Please enter a research topic.")
    else:
        try:
            # Create a  placeholder for the final report
            report_placeholder = st.empty()

            # Run the multi-agent Research Process
            enhanced_report = asyncio.run(run_research_process(research_topic))

            # Display the final enhanced report
            report_placeholder.markdown(
                "## Enhanced Research Report (Multi-Agent Output)"
            )
            report_placeholder.markdown(enhanced_report)

            # Add a download button for the report

            st.download_button(
                "Download Report",
                enhanced_report,
                file_name=f"{research_topic.replace(' ', '_')}_report.md",
                mime='text/markdown',
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown('---')
st.markdown("Powered by OpenAI Agents SDK and FireCrawler - Your own deep"
            "research solution with subscripton fees ")
