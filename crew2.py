import os
import yaml
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

os.environ["SERPER_API_KEY"] = "bb2b64f8be808b215b6cc8f9785cb52b7ff973ed"
os.environ["OPENAI_API_KEY"] = "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"

serper_tool = SerperDevTool()

agent_memory = {}

def update_memory(agent_id, context):
    if agent_id in agent_memory:
        agent_memory[agent_id].append(context)
    else:
        agent_memory[agent_id] = [context]

with open(r'C:\Users\Yatharth\Desktop\desktop1\AI\Prasunethon\crew2\agents.yaml', 'r') as file:
    agents_config = yaml.safe_load(file)

with open(r'C:\Users\Yatharth\Desktop\desktop1\AI\Prasunethon\crew2\tasks.yaml', 'r') as file:
    tasks_config = yaml.safe_load(file)

symptom_inquiry_agent = Agent(
    role=agents_config['symptom_inquiry_agent']['role'],
    goal=agents_config['symptom_inquiry_agent']['goal'],
    verbose=True,
    memory=agents_config['symptom_inquiry_agent']['memory'],
    backstory=agents_config['symptom_inquiry_agent']['backstory'],
    tools=[serper_tool]
)

medical_insight_agent = Agent(
    role=agents_config['medical_insight_agent']['role'],
    goal=agents_config['medical_insight_agent']['goal'],
    verbose=True,
    memory=agents_config['medical_insight_agent']['memory'],
    backstory=agents_config['medical_insight_agent']['backstory'],
    tools=[serper_tool]
)

medication_advisor_agent = Agent(
    role=agents_config['medication_advisor_agent']['role'],
    goal=agents_config['medication_advisor_agent']['goal'],
    verbose=True,
    memory=agents_config['medication_advisor_agent']['memory'],
    backstory=agents_config['medication_advisor_agent']['backstory'],
    tools=[serper_tool]
)

doctor_intervention_advisor_agent = Agent(
    role=agents_config['doctor_intervention_advisor_agent']['role'],
    goal=agents_config['doctor_intervention_advisor_agent']['goal'],
    verbose=True,
    memory=agents_config['doctor_intervention_advisor_agent']['memory'],
    backstory=agents_config['doctor_intervention_advisor_agent']['backstory'],
    tools=[]
)

collect_symptoms_queries_task = Task(
    human_input=True,
    description=tasks_config['collect_symptoms_queries_task']['description'],
    expected_output=tasks_config['collect_symptoms_queries_task']['expected_output'],
    tools=[serper_tool],
    agent=symptom_inquiry_agent
)

provide_medical_insights_task = Task(
    description=tasks_config['provide_medical_insights_task']['description'],
    expected_output=tasks_config['provide_medical_insights_task']['expected_output'],
    tools=[serper_tool],
    context=[collect_symptoms_queries_task],
    agent=medical_insight_agent
)

suggest_medication_usage_task = Task(
    description=tasks_config['suggest_medication_usage_task']['description'],
    expected_output=tasks_config['suggest_medication_usage_task']['expected_output'],
    tools=[serper_tool],
    context=[collect_symptoms_queries_task, provide_medical_insights_task],
    agent=medication_advisor_agent
)

advise_professional_help_task = Task(
    description=tasks_config['advise_professional_help_task']['description'],
    expected_output=tasks_config['advise_professional_help_task']['expected_output'],
    tools=[],
    context=[collect_symptoms_queries_task, provide_medical_insights_task],
    agent=doctor_intervention_advisor_agent
)

manager = Agent(
    role="Manager",
    goal="Manage the crew and ensure the tasks are completed efficiently.",
    backstory="You're an experienced manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard, especially adhering to the provided input",
    allow_delegation=False,
)

virtual_nursing_assistant_crew = Crew(
    agents=[symptom_inquiry_agent, medical_insight_agent, medication_advisor_agent, doctor_intervention_advisor_agent],
    tasks=[collect_symptoms_queries_task, provide_medical_insights_task, suggest_medication_usage_task, advise_professional_help_task],
    process=Process.hierarchical,
    manager_agent=manager
)

user_symptoms = input("Please describe your symptoms: ")

inputs = {'symptoms': user_symptoms,
          'specific_requirements': 'Focus on the symptoms and only according to the symptoms provided suggest what measures should the user take, what medicine should he get prescription for and if it is serious inform him that he needs to seek professional help; and if no professional help is required then just say all is well',
          'time_frame': 'last five years',
          'data_sources': ['articles, online medical forums, wikipedia'],
          'output_format': 'comprehensive report'}

result = virtual_nursing_assistant_crew.kickoff(inputs=inputs)

print(result)
