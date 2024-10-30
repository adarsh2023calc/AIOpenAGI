import time
import os
from openagi.agent import Admin
from openagi.worker import Worker
from openagi.actions.files import WriteFileAction
from openagi.actions.tools.ddg_search import DuckDuckGoNewsSearch
from openagi.actions.tools.webloader import WebBaseContextTool
from openagi.llms.openai import OpenAIModel
from openagi.memory import Memory
from openagi.actions.tools.tavilyqasearch import TavilyWebSearchQA
from openagi.planner.task_decomposer import TaskPlanner

# Set environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('SECRET_KEY')
os.environ['TAVILY_API_KEY'] = "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"

# Load OpenAI configuration
config = OpenAIModel.load_from_env_config()
llm = OpenAIModel(config=config)

class Agent:
    def __init__(self, research_question, writer_question, reviewer_question, 
                 timeout=30, max_retries=5, backoff_factor=2, cache={}):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.cache = cache  # Caching responses to avoid redundant requests

        # Initialize Admin and memory with persistent storage
        self.admin = Admin(
            llm=llm,
            actions=[TavilyWebSearchQA],
            planner=TaskPlanner(human_intervene=False),
            memory=Memory()
        )

        self.researcher = Worker(
            role="Researcher",
            instructions=research_question,
            actions=[
                TavilyWebSearchQA,
                WebBaseContextTool,
            ],
        )

        self.writer = Worker(
            role="Writer",
            instructions=writer_question,
            actions=[
                TavilyWebSearchQA,
                WebBaseContextTool,
            ],
        )

        self.reviewer = Worker(
            role="Reviewer",
            instructions=reviewer_question,
            actions=[
                WebBaseContextTool,
                WriteFileAction,
            ],
        )

    def run_admin_task(self, query, description):
        # Check cache first to avoid redundant requests
        if query in self.cache:
            print("Fetching result from cache.")
            return self.cache[query]

        # Try to run task with exponential backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.admin.run(query=query, description=description)
                self.cache[query] = result  # Cache the result
                return result

            except TimeoutError as e:
                print(f"Attempt {attempt}: TimeoutError - {e}. Retrying in {self.backoff_factor ** attempt} seconds.")
            except Exception as e:
                print(f"Attempt {attempt}: Error - {e}. Retrying in {self.backoff_factor ** attempt} seconds.")

            time.sleep(self.backoff_factor ** attempt)

        print("Max retries reached. Task failed.")
        return None

if __name__ == "__main__":
    company_name = input("Enter a company name: ")

    # Define questions and description
    research_question = f"""Research the industry of {company_name}.
                            Identify the key offerings and strategic focus areas of {company_name}. Only Top 5 results"""

    write_question = f"""Analyze trends and standards in {company_name}’s sector related to AI, ML, and automation.
                         Propose relevant use cases for AI and ML in {company_name}. Only Top 5 results"""

    review_question = """Review the content in a presentation-friendly format."""

    description = f"""Create an engaging research proposal on GenAI & ML Use Cases for {company_name}.
                      Structure with introduction, body, and conclusion, and save as a PDF."""

    # Initialize Agent with caching and retry settings
    agent = Agent(research_question=research_question,
                  reviewer_question=review_question,
                  writer_question=write_question,
                  timeout=30, max_retries=5, backoff_factor=2)

    # Run the task with caching to reduce redundant API calls
    result = agent.run_admin_task(query="Write a research proposal",
                                  description=f"GenAI & ML Use Cases for {company_name}")

    if result:
        print(result)
    else:
        print("The research proposal generation failed after all retry attempts.")
