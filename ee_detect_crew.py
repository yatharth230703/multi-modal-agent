import os
import yaml
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from eye_disease_detection_tool import EyeDiseaseDetectionTool

os.environ["SERPER_API_KEY"] = "bb2b64f8be808b215b6cc8f9785cb52b7ff973ed"
os.environ["OPENAI_API_KEY"] = "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"

serper_tool = SerperDevTool()
eye_disease_tool = EyeDiseaseDetectionTool(model_path=r'C:\Users\Yatharth\Desktop\desktop1\AI\Prasunethon\eye_disease_detect\model_eye.pkl')

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

eye_disease_detection_agent = Agent(
    role=agents_config['eye_disease_detection_agent']['role'],
    goal=agents_config['eye_disease_detection_agent']['goal'],
    verbose=True,
    memory=agents_config['eye_disease_detection_agent']['memory'],
    backstory=agents_config['eye_disease_detection_agent']['backstory'],
    tools=[eye_disease_tool]
)

identify_eye_disease_task = Task(
    human_input=True,
    description=tasks_config['identify_eye_disease_task']['description'],
    expected_output=tasks_config['identify_eye_disease_task']['expected_output'],
    tools=[eye_disease_tool],
    agent=eye_disease_detection_agent
)

manager = Agent(
    role="Manager",
    goal="Manage the crew and ensure the tasks are completed efficiently.",
    backstory="You're an experienced manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
    allow_delegation=False,
)

virtual_nursing_assistant_crew = Crew(
    agents=[symptom_inquiry_agent, eye_disease_detection_agent],
    tasks=[identify_eye_disease_task],
    process=Process.sequential,
    manager_agent=manager
)

image_path = r'C:\Users\Yatharth\Desktop\desktop1\AI\Prasunethon\eye_disease_detect\glaucoma.png'
inputs = {'image_path': image_path}

result = virtual_nursing_assistant_crew.kickoff(inputs=inputs)

print(result)
