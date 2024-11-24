# crew.py
 
import os
import openai
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
# from src.esg.my_llm import MyLLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, ScrapeElementFromWebsiteTool, WebsiteSearchTool, PDFSearchTool
from langchain_openai import ChatOpenAI


load_dotenv()

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

openai.api_key = api_key

# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)

# Uncomment the following line to use an example of a custom tool
# from esg.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

@CrewBase
class Esg():
	"""Esg crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def pesquisa_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['pesquisa_agent'],
			tools=[SerperDevTool(), ScrapeWebsiteTool(), WebsiteSearchTool()],
			verbose=True,
			memory=True,
			allow_delegation=True,
			allow_interruption=True,  # Permite interrupções para reagir rapidamente a mudanças nos produtos dos concorrentes
			allow_fallback=True,
			llm=llm

		)

	@agent
	def levantamento_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['levantamento_agent'],
			tools=[SerperDevTool(), ScrapeElementFromWebsiteTool(), PDFSearchTool()],
			verbose=True,
			memory=True,
			allow_delegation=True,
			allow_interruption=True,  # Permite interrupções para reagir rapidamente a mudanças nos produtos dos concorrentes
			allow_fallback=True,
			llm=llm
		)

	@agent
	def sugestao_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['sugestao_agent'],
			tools=[SerperDevTool(), WebsiteSearchTool()],
			verbose=True,
			memory=True,
			allow_delegation=True,
			allow_interruption=True,  # Permite interrupções para reagir rapidamente a mudanças nos produtos dos concorrentes
			allow_fallback=True,
			llm=llm
		)

	@agent
	def conformidade_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['conformidade_agent'],
			tools=[PDFSearchTool(), WebsiteSearchTool()],
			verbose=True,
			memory=True,
			allow_delegation=True,
			allow_interruption=True,  # Permite interrupções para reagir rapidamente a mudanças nos produtos dos concorrentes
			allow_fallback=True,
			llm=llm
		)

	@agent
	def planejamento_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['planejamento_agent'],
			tools=[],  # No specific tools needed, but can be added if necessary
			verbose=True,
			memory=True,
			allow_delegation=True,
			allow_interruption=True,  # Permite interrupções para reagir rapidamente a mudanças nos produtos dos concorrentes
			allow_fallback=True,
			llm=llm
		)

	@task
	def pesquisa_mercado_task(self) -> Task:
		return Task(
			config=self.tasks_config['pesquisa_mercado_task'],
			output_file='pesquisa_empresa.md',
			guardrails=[{"output_format": "markdown"}, {"max_length": 15000}]
		)

	@task
	def levantamento_esg_task(self) -> Task:
		return Task(
			config=self.tasks_config['levantamento_esg_task'],
			output_file='levantamento_esg.md',
			guardrails=[{"output_format": "markdown"}, {"max_length": 15000}]
		)

	@task
	def sugestao_esg_task(self) -> Task:
		return Task(
			config=self.tasks_config['sugestao_esg_task'],
			output_file='sugestao_esg.md',
			guardrails=[{"output_format": "markdown"}, {"max_length": 15000}]
		)

	@task
	def conformidade_task(self) -> Task:
		return Task(
			config=self.tasks_config['conformidade_task'],
			output_file='conformidade_relatorio.md',
			guardrails=[{"output_format": "markdown"}, {"max_length": 15000}]
		)

	@task
	def planejamento_task(self) -> Task:
		return Task(
			config=self.tasks_config['planejamento_task'],
			output_file='plano_implementacao.md',
			guardrails=[{"output_format": "markdown"}, {"max_length": 15000}]
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Esg crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
