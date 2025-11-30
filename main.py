
import os
import logging
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any
from openai import OpenAI
import wikipedia
from unidecode import unidecode
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)


class Agent:
    def __init__(self, client: OpenAI, model = "mistral"):
        self.client = client
        self.__max_iterations = 5
        self.__model = model
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Agent initialized with model: {model}")

    def __tools(self):
        self.logger.debug("Providing tool definitions to LLM.")
        return [
            {
                "type": "function",
                "name": "get_public_ip",
                "description": "Get the public IP address of the user.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "type": "function",
                "name": "get_location",
                "description": "Get city location based on public user IP address.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ip_address": {
                            "type": "string",
                            "description": "The public IP address of the user."
                        }
                    },
                    "required": ["ip_address"]
                }
            },
            {
                "type": "function",
                "name": "get_location_info",
                "description": "Get facts about city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the city."
                        }
                    },
                    "required": ["name"]
                }
            }
        ]
    
    def __execute_tool(self, tool_call: Any) -> str:
        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments
        self.logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                self.logger.warning("Tool arguments could not be parsed as JSON, using as-is.")
        try:
            if tool_name == "get_public_ip":
                return self.__integrations_get_public_ip()
            elif tool_name == "get_location":
                ip_address = arguments.get("ip_address")
                return self.__integrations_get_location(ip_address)
            elif tool_name == "get_location_info":
                name = arguments.get("name")
                return self.__integrations_get_location_facts(name)
            else:
                self.logger.warning(f"Unknown tool requested by LLM: {tool_name}. Ignoring and continuing.")
                return f"Tool '{tool_name}' is not available."
        except Exception as e:
            self.logger.exception(f"Error executing tool {tool_name}: {e}")
            raise
        
        
    def __integrations_get_public_ip(self) -> str:
        self.logger.info("Calling external service to get public IP.")
        response = requests.get("https://ifconfig.me/ip")
        self.logger.debug(f"Public IP response: {response.text.strip()}")
        return response.text.strip()
    
    def __integrations_get_location(self, ip_address: str) -> str:
        self.logger.info(f"Calling external service to get location info for IP: {ip_address}")
        response = requests.get(f"http://ip-api.com/json/{ip_address}")
        self.logger.debug(f"Location info response: {response.text.strip()}")
        return response.text.strip()
    
    def __integrations_get_location_facts(self, name: str) -> str:
        self.logger.info(f"Fetching location facts for: {name}")
        try:
            page = self.__resolve_city_page(name)
            if page is None:
                return json.dumps({"error": "not_found", "requested": name})
            content = page.content
            excerpt = content.split('\n\n')[0].strip()
            if len(excerpt) > 500:
                excerpt = excerpt[:500].rsplit(' ', 1)[0] + '…'
            payload = {
                "requested": name,
                "resolved_title": page.title,
                "url": page.url,
                "excerpt": excerpt,
            }
            self.logger.debug(f"Location facts payload: {payload}")
            return json.dumps(payload)
        except wikipedia.exceptions.DisambiguationError as e:
            self.logger.error(f"Disambiguation error for {name}: {e}")
            return json.dumps({"error": "disambiguation", "requested": name, "options": e.options[:10]})
        except wikipedia.exceptions.PageError as e:
            self.logger.error(f"Page error for {name}: {e}")
            return json.dumps({"error": "not_found", "requested": name})
        except Exception as e:
            self.logger.exception(f"Error fetching location facts for {name}: {e}")
            return json.dumps({"error": "general", "requested": name, "details": str(e)})

    def __resolve_city_page(self, name: str):
        """Resolve a city Wikipedia page using transliteration, alias generation, and scoring of search results."""
        raw = name.strip()
        ascii_name = unidecode(raw)
        candidates = list(dict.fromkeys([
            raw,
            raw.title(),
            ascii_name,
            ascii_name.title(),
            raw.replace(' ', '-'),
            ascii_name.replace(' ', '-'),
        ]))
        for cand in candidates:
            try:
                return wikipedia.page(cand, auto_suggest=False)
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                continue
            except Exception:
                continue
        search_query = ascii_name
        results = wikipedia.search(search_query)
        if not results:
            return None
        self.logger.debug(f"Search results for '{search_query}': {results}")
        best_page = None
        best_score = -1
        for r in results[:10]:
            try:
                page = wikipedia.page(r, auto_suggest=True)
            except Exception:
                continue
            title_lower = page.title.lower()
            score = 0
            if title_lower == raw.lower() or title_lower == ascii_name.lower():
                score += 100
            if raw.lower() in title_lower:
                score += 40
            if 'city' in title_lower:
                score += 15
            try:
                cats = [c.lower() for c in page.categories]
                if any('cities' in c or 'populated places' in c for c in cats):
                    score += 25
            except Exception:
                pass
            if score > best_score:
                best_score = score
                best_page = page
        return best_page

    def __summarize_location_facts(self, content: str) -> str:
        """Summarize JSON payload (from location facts) or error into user-friendly text."""
        if not content:
            return "No information was retrieved."
        # Try JSON parse
        try:
            data = json.loads(content)
        except Exception:
            return content[:500]
        if 'error' in data:
            err = data['error']
            if err == 'disambiguation':
                opts = ', '.join(data.get('options', [])[:5])
                return f"Multiple possible matches: {opts}. Please specify more details (country/region)."
            if err == 'not_found':
                return f"I could not find a page for '{data.get('requested')}'. Please check spelling or add country." 
            return f"Error fetching facts ({err}). Try again later." 
        title = data.get('resolved_title', data.get('requested','Unknown'))
        excerpt = data.get('excerpt','')
        url = data.get('url','')
        return f"{title}: {excerpt} Source: {url}"[:600]
    
    def run(self, messages: List[Dict[str, Any]]):
        self.logger.info("Starting agent run loop.")
        iteration = 0
        while iteration < self.__max_iterations:
            iteration += 1
            self.logger.info(f"Iteration {iteration} - sending messages to LLM.")
            response = self.client.chat.completions.create(
                model=self.__model,
                messages=messages,
                tools=self.__tools(),
                tool_choice="auto"
            )
            response_message = response.choices[0].message
            self.logger.debug(f"LLM response: {str(response_message)}")
            if response_message.tool_calls:
                tool_names = [tc.function.name for tc in response_message.tool_calls]
                self.logger.info(f"Tool calls detected: {tool_names}")
                messages.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in response_message.tool_calls
                    ],
                })
                for tool_call in response_message.tool_calls:
                    tool_response = self.__execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_response,
                    })
                    if tool_call.function.name == "get_location_info":
                        summary = self.__summarize_location_facts(tool_response)
                        self.logger.info("Finalizing after get_location_info.")
                        messages.append({"role": "assistant", "content": summary})
                        return summary
                continue
            final_content = response_message.content
            self.logger.info("Final response from LLM received.")
            messages.append({"role": "assistant", "content": final_content})
            return final_content
        self.logger.error("Max iterations reached without final response.")
        raise Exception("Max iterations reached without final response.")
                
                
                

def main():
    load_dotenv()
    logging.info("Creating OpenAI client.")
    client = OpenAI(
        base_url=os.environ.get("OPENAI_API_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    logging.info("Instantiating Agent.")
    agent = Agent(client=client, model="ollama/llama3.2")
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. TOOL USAGE RULES:\n"
                "1. If the user asks for THEIR location (e.g. 'Where am I', 'What city am I in', 'Facts about my location') and has NOT provided a city name: FIRST call get_public_ip, THEN call get_location(ip_address=<returned IP>) to obtain the city name. Optionally then call get_location_info(name=<city>) exactly once to fetch facts.\n"
                "2. If the user directly provides a city name (e.g. 'Tell me facts about Prague'), SKIP get_public_ip and get_location and call ONLY get_location_info(name=<city>) once.\n"
                "3. Never call the same tool more than once for the same purpose. Do not invent tools.\n"
                "4. After tool(s) finish, produce a concise paragraph summary and STOP calling tools.\n"
                "5. If a tool returns an error/disambiguation message, explain it and request clarification instead of calling more tools."
            ),
        },
        {"role": "user", "content": "Tell me some facts about my location?"},
    ]
    
    logging.info("Running agent with initial messages.")
    response = agent.run(messages=messages)
    logging.info("Agent run complete. Outputting result.")
    print("Info about my location: ")
    print(response)

if __name__ == "__main__":
    main()
