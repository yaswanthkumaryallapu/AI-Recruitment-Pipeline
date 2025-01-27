import random
import pandas as pd
from faker import Faker
from together import Together
import os

# Set Together API key
os.environ["TOGETHER_API_KEY"] = "Your_api_key"
client = Together()

# Initialize Faker instance
fake = Faker()

# Define roles and their associated skillsets
roles = {
    "Data Scientist": ["Python", "Statistics", "Machine Learning", "Deep Learning", "SQL", "Data Visualization", "TensorFlow"],
    "Data Engineer": ["MLOps", "Airflow", "ETL", "Big Data", "Cloud Platforms", "Spark", "Data Warehousing"],
    "Software Engineer": ["Java", "C++", "Python", "Algorithms", "Data Structures", "System Design", "DevOps"],
    "Product Manager": ["Roadmaps", "Stakeholder Communication", "Agile", "Data Analysis", "Leadership", "Market Research"],
    "UI Engineer": ["HTML", "CSS", "JavaScript", "React", "UI/UX Design", "Wireframing"],
    "Cybersecurity Analyst": ["Risk Assessment", "Penetration Testing", "Network Security", "SIEM", "Incident Response", "Encryption"],
    "Cloud Architect": ["AWS", "Azure", "GCP", "Terraform", "Kubernetes", "Cloud Security"],
    "Business Analyst": ["Requirement Gathering", "Data Analysis", "SQL", "Presentation", "Problem-Solving", "Communication"],
    "AI Researcher": ["NLP", "GANs", "Transformers", "Reinforcement Learning", "Computer Vision", "PyTorch"],
    "DevOps Engineer": ["CI/CD", "Docker", "Kubernetes", "Ansible", "Infrastructure as Code", "Monitoring Tools", "Cloud Platforms"],
    "Database Administrator": ["SQL", "NoSQL", "Database Optimization", "Backup and Recovery", "Database Security", "Performance Tuning"],
    "Mobile App Developer": ["Swift", "Kotlin", "Flutter", "React Native", "Mobile UI/UX", "API Integration"],
    "Game Developer": ["Unity", "Unreal Engine", "C#", "3D Modeling", "Game Physics", "Gameplay Programming"],
    "Machine Learning Engineer": ["Python", "Scikit-learn", "TensorFlow", "PyTorch", "Feature Engineering", "Hyperparameter Tuning"],
    "Blockchain Developer": ["Solidity", "Ethereum", "Smart Contracts", "Cryptography", "Web3.js", "Consensus Algorithms"],
    "Digital Marketing Specialist": ["SEO", "Google Ads", "Content Marketing", "Social Media Marketing", "Email Marketing", "Analytics Tools"],
    "Full Stack Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Database Management", "APIs"],
    "QA Engineer": ["Manual Testing", "Automated Testing", "Selenium", "Bug Tracking", "Performance Testing", "Test Case Development"],
    "Robotics Engineer": ["ROS", "Control Systems", "Mechatronics", "Sensor Fusion", "Path Planning", "MATLAB"],
    "System Administrator": ["Linux/Unix", "Windows Server", "Active Directory", "Networking", "Virtualization", "System Monitoring"],
    "Cloud Engineer": ["Cloud Migration", "Cloud Cost Optimization", "Serverless Architecture", "Cloud Networking", "Scripting"],
    "Content Writer": ["SEO Writing", "Copywriting", "Technical Writing", "Creative Writing", "Proofreading", "Content Strategy"],
    "Data Analyst": ["Excel", "Python", "Tableau", "Power BI", "SQL", "Statistical Analysis"],
    "Human Resources Specialist": ["Recruitment", "Employee Relations", "HR Software", "Onboarding", "Training and Development", "Conflict Resolution"],
    "IT Support Specialist": ["Troubleshooting", "Technical Support", "Hardware/Software Knowledge", "Networking", "Ticketing Systems"],
    "UX Designer": ["User Research", "Prototyping", "Wireframing", "Usability Testing", "Design Tools", "Interaction Design"],
    "E-commerce Specialist": ["Product Listing", "Inventory Management", "SEO for E-commerce", "Online Advertising", "Customer Service", "Analytics"],
    "AR/VR Developer": ["Unity", "C#", "3D Modeling", "Oculus SDK", "Augmented Reality Markers", "Spatial Computing"],
    "Data Architect": ["Data Modeling", "Data Integration", "Big Data", "Cloud Data Solutions", "ETL Tools", "Database Design"]
}


# Experience levels and work environments
experience_levels = ["Entry-level", "Mid-level", "Senior-level", "Lead", "Director"]
work_environments = ["Remote", "Hybrid", "In-office"]

# Job description templates
job_description_variations = [
    "Join our team as a Product Manager and leverage your {years} years of experience to make an impact and contribute to innovative projects.",
    "We are looking for an experienced {role} to join our team and help drive groundbreaking solutions in {field}.",
    "As a {role}, you'll lead the design and development of cutting-edge {technologies}.",
    "We're seeking a talented {role} to work on {specific_project} and bring new ideas to life.",
    "Join our fast-growing team and help us scale our product offerings as a {role} with expertise in {skillset}.",
    "As a {role}, you will play a pivotal role in shaping the future of {industry}.",
    "Looking for an experienced {role} to join us in driving strategic initiatives and bringing innovation to {industry}.",
    "We need a {role} to enhance our team's technical capabilities and contribute to solving complex {challenges}.",
    "Help us build the next-generation products as a {role} and work with a dynamic, cross-functional team.",
    "If you're passionate about {field}, we need your expertise to help us deliver impactful products as a {role}.",
    "We're hiring a {role} to develop and deliver high-quality solutions to transform our {industry}.",
    "Take the lead in driving innovation as a {role} in a collaborative environment focused on excellence in {field}.",
    "Be part of a passionate team at the forefront of {technologies} as a {role}, delivering solutions that shape the future."
]

# Function to call Together API
def call_together_api(prompt):
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{"role": "user", "content": f"{prompt}"}],
    )
    content = response.choices[0].message.content
    return content.replace("**", "")  # Remove bold formatting markers

# Shortened reasons for decision based on performance
reasons_for_rejection = [
    "Lacks hands-on experience with cloud platforms.",
    "Insufficient system design expertise for senior role.",
    "Lacked leadership skills for a senior position.",
    "No experience in back-end development.",
    "Needs improvement in machine learning algorithms."
]

reasons_for_selection = [
    "Strong technical skills in AI and ML.",
    "Impressive leadership and communication abilities.",
    "Perfectly aligned with data engineering needs.",
    "Excellent full-stack development experience.",
    "Solid experience in machine learning and AI."
]

# Function to generate candidate data
def generate_candidate_data(n):
    data = []
    for _ in range(n):
        # Randomly generate candidate details
        name = fake.name()
        first_name, last_name = name.split(" ", 1)
        role = random.choice(list(roles.keys()))
        skillset = roles[role]
        poor_skills = random.sample(skillset, random.randint(1, min(3, len(skillset))))

        # Generate interview prompt to send to Together API (Use "Interviewer" instead of a name)
        transcript_prompt = f"Simulate a professional interview for a candidate named {name} applying for the role of {role}. The interview should cover topics like {', '.join(poor_skills)} and include the greeting, questions, answers, and polite closing at the end. Make the responses realistic with natural pauses like 'um...' or 'uh...', and reflect the candidate’s abilities and experience. Use 'Interviewer' instead of any specific name."

        # Get the interview transcript from Together API
        transcript = call_together_api(transcript_prompt)

        # Generate resume using Together API
        resume_prompt = f"Create a professional resume for a candidate named {name}, skilled in {', '.join([skill for skill in skillset if skill not in poor_skills])}, applying for the role of {role}. Include professional summaries, achievements, and a clear structure."
        resume = call_together_api(resume_prompt)

        # Randomly select a job description variation
        job_description = random.choice(job_description_variations).format(
            years=random.randint(5, 10),
            role=role,
            field=random.choice(["data science", "software engineering", "AI research", "cloud technologies"]),
            skillset=", ".join(random.sample(skillset, 3)),
            technologies=random.choice(["AI", "machine learning", "cloud computing", "data analysis"]),
            industry=random.choice(["finance", "healthcare", "e-commerce", "education"]),
            specific_project=random.choice(["AI model development", "cloud migration", "enterprise software development"]),
            challenges=random.choice(["complex problems", "scalable solutions", "business challenges"])
        )

        resume_skills = random.sample(skillset, random.randint(1, len(skillset)))  # Randomly pick resume skills

        # Simulate mismatch by flipping a coin (50% chance of rejection if skills match)
        if set(resume_skills).intersection(skillset) and random.random() > 0.5:
            outcome = "select"
            reason = random.choice(reasons_for_selection)
        else:
            outcome = "reject"
            reason = random.choice(reasons_for_rejection)


        # Append data for DataFrame
        candidate_id = first_name[:4].lower() + last_name[:2].lower() + str(random.randint(100, 999))
        data.append({
            "ID": candidate_id,
            "Name": name,
            "Role": role,
            "Transcript": transcript,
            "Resume": resume,
            "decision": outcome,
            "Reason for decision": reason,
            "Job Description": job_description
        })
    return pd.DataFrame(data)

# Generate data for n candidates
n = int(input("Enter the number of candidates: "))
data = generate_candidate_data(n)

# Save to Excel
excel_filename = "datasets/data1.xlsx"
data.to_excel(excel_filename, index=False)
print(f"Data generated and saved to '{excel_filename}'.")
