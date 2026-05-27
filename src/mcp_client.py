import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

async def call_mcp_search_async(query: str, max_results: int = 5) -> Optional[List[Dict[str, Any]]]:
    """
    Asynchronously spawns and connects to the DuckDuckGo MCP server via stdio transport,
    calls the 'search' tool, and parses the returned search results.
    
    Args:
        query: The search query string.
        max_results: The maximum number of results to fetch.
        
    Returns:
        A list of search result dictionaries containing 'title', 'body', and 'href' keys,
        or None if the server is unavailable or fails.
    """
    # Locate the duckduckgo-mcp-server executable in the current virtual environment to be self-contained
    python_exe = Path(sys.executable)
    
    # On Windows, pip creates entry points in .venv/Scripts/
    # On Unix, they are in .venv/bin/
    mcp_server_exe = python_exe.parent / "duckduckgo-mcp-server"
    if sys.platform.startswith("win"):
        mcp_server_exe = python_exe.parent / "duckduckgo-mcp-server.exe"
        
    command = str(mcp_server_exe)
    args = []
    
    # Fallback checks if the executable is not in the virtual environment Scripts/bin folder
    if not mcp_server_exe.exists():
        logger.warning(f"MCP server executable not found at '{mcp_server_exe}'. Falling back to global/PATH search.")
        command = "duckduckgo-mcp-server"
        
    # We can also fall back to running via python module if needed
    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=dict(os.environ)  # Pass current environment variables
    )
    
    logger.info(f"Spawning DuckDuckGo MCP server using command: '{command}'")
    
    try:
        # Connect to the server via stdio
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the MCP session
                await session.initialize()
                
                # Retrieve available tools
                tools_response = await session.list_tools()
                available_tools = [t.name for t in tools_response.tools]
                logger.info(f"Connected to MCP Server. Available tools: {available_tools}")
                
                # Check if the 'search' tool is supported
                if "search" not in available_tools:
                    logger.error("DuckDuckGo MCP server does not expose a 'search' tool.")
                    return None
                    
                # Call the 'search' tool with query parameters
                # duckduckgo-mcp-server search takes a 'query' argument, and optionally 'max_results'
                logger.info(f"Calling MCP search tool for query: '{query}'")
                tool_result = await session.call_tool(
                    "search",
                    arguments={"query": query, "max_results": max_results}
                )
                
                # Parse results from the MCP tool response
                results = []
                if tool_result and hasattr(tool_result, "content"):
                    for block in tool_result.content:
                        if hasattr(block, "text") and block.text:
                            text_content = block.text.strip()
                            
                            # The response is usually a JSON string containing search results, or a Markdown string.
                            # Try to parse as JSON first
                            try:
                                parsed = json.loads(text_content)
                                if isinstance(parsed, list):
                                    for item in parsed:
                                        results.append({
                                            "title": item.get("title") or item.get("name") or "No Title",
                                            "body": item.get("description") or item.get("body") or item.get("snippet") or "No content snippet.",
                                            "href": item.get("href") or item.get("url") or ""
                                        })
                                elif isinstance(parsed, dict):
                                    # Check if results are wrapped under a key like 'results' or 'data'
                                    items = parsed.get("results") or parsed.get("data") or [parsed]
                                    if isinstance(items, list):
                                        for item in items:
                                            results.append({
                                                "title": item.get("title") or item.get("name") or "No Title",
                                                "body": item.get("description") or item.get("body") or item.get("snippet") or "No content snippet.",
                                                "href": item.get("href") or item.get("url") or ""
                                            })
                            except (json.JSONDecodeError, TypeError):
                                # If it's not JSON, it is likely standard markdown/text output.
                                # We can parse the text into structured elements.
                                import re
                                try:
                                    lines = text_content.split('\n')
                                    current_result = {}
                                    parsed_results = []
                                    
                                    for line in lines:
                                        line_stripped = line.strip()
                                        if not line_stripped:
                                            continue
                                            
                                        # Match "<number>. <Title>"
                                        match_title = re.match(r'^\d+\.\s+(.+)$', line_stripped)
                                        if match_title:
                                            if current_result:
                                                parsed_results.append(current_result)
                                            current_result = {
                                                "title": match_title.group(1).strip(),
                                                "body": "",
                                                "href": ""
                                            }
                                            continue
                                            
                                        if line_stripped.startswith("URL:"):
                                            if current_result:
                                                current_result["href"] = line_stripped[4:].strip()
                                            continue
                                            
                                        if line_stripped.startswith("Summary:"):
                                            if current_result:
                                                current_result["body"] = line_stripped[8:].strip()
                                            continue
                                            
                                        # Append to body if it's additional text under a result
                                        if current_result and not line_stripped.startswith("Found"):
                                            if current_result["body"]:
                                                current_result["body"] += " " + line_stripped
                                            else:
                                                current_result["body"] = line_stripped
                                                
                                    if current_result:
                                        parsed_results.append(current_result)
                                        
                                    if parsed_results:
                                        results.extend(parsed_results)
                                except Exception as parse_err:
                                    logger.warning(f"Error parsing raw MCP text search results: {str(parse_err)}")
                                
                            # If parsing as structured JSON or text yielded no results, return the raw text structured nicely
                            if not results:
                                results.append({
                                    "title": f"Web Search Results for: {query}",
                                    "body": text_content,
                                    "href": "MCP Server"
                                })
                
                return results if results else None
                
    except Exception as e:
        logger.error(f"Error during DuckDuckGo MCP server communication: {str(e)}", exc_info=True)
        return None

def search_via_mcp(query: str, max_results: int = 5) -> Optional[List[Dict[str, Any]]]:
    """
    Synchronous wrapper around call_mcp_search_async to be called from synchronous code.
    Handles event loops cleanly.
    """
    try:
        # Check if an event loop is running (FastAPI threadpools usually don't have one running)
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        # In case we're inside a running loop (like an async endpoint), we must run in a separate thread/executor
        # to avoid "this event loop is already running" error, or use a runner that schedules it.
        # But for FastAPI's sync routes (run in a threadpool) this is not active.
        # To be safe:
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(call_mcp_search_async(query, max_results)))
            return future.result()
    else:
        return asyncio.run(call_mcp_search_async(query, max_results))
