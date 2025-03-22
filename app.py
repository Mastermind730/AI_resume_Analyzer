import io
import os
import json
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import spacy
import re

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("You may need to install it with: python -m spacy download en_core_web_sm")
    exit(1)

def pdf_reader(file):
    try:
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)

        with open(file, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)
            
            text = fake_file_handle.getvalue()

        # Close open handles
        converter.close()
        fake_file_handle.close()

        return text
    except FileNotFoundError:
        print(f"Error: The file {file} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_info_custom(text):
    """Custom implementation to extract resume information without relying on pyresparser"""
    doc = nlp(text)
    
    # Initialize data dictionary
    data = {
        "name": "",
        "email": "",
        "phone": "",
        "skills": []
    }
    
    # Extract email using regex
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        data["email"] = emails[0]
    
    # Extract phone using regex
    phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phones = re.findall(phone_pattern, text)
    if phones:
        data["phone"] = phones[0]
    
    # Extract potential skills (all proper nouns and technical terms)
    # This is a simple approach - in a real system you'd have a predefined skills list
    skill_keywords = ["python", "java", "javascript", "html", "css", "react", "node", 
                      "sql", "mongodb", "aws", "azure", "docker", "kubernetes", "machine learning",
                      "data analysis", "excel", "powerbi", "tableau", "r", "c++", "c#",
                      "git", "agile", "scrum", "project management"]
    
    # Get skills from predefined list
    found_skills = []
    text_lower = text.lower()
    for skill in skill_keywords:
        if skill in text_lower:
            found_skills.append(skill.title())  # Capitalize for consistency
    
    data["skills"] = found_skills
    
    # Try to extract name (first few lines often contain the name)
    lines = text.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if line and len(line) < 40 and not any(c in line for c in "@.,:;()[]{}"):
            data["name"] = line
            break
    
    return data

def match_skills(resume_skills, job_requirement):
    """Match skills between resume and job requirements"""
    # Convert to lowercase for case-insensitive matching
    resume_skills_lower = [skill.lower() for skill in resume_skills]
    job_requirement_lower = [skill.lower() for skill in job_requirement]
    
    # Find matches
    matched_skills = []
    for skill in job_requirement_lower:
        for resume_skill in resume_skills_lower:
            if skill in resume_skill or resume_skill in skill:
                matched_skills.append(skill)
                break
    
    return matched_skills

def score_resume(resume_data, job_data):
    """Score the resume based on job data"""
    if not resume_data or not job_data:
        return 0
    
    score = 0
    if resume_data.get('skills') and job_data.get('required_skills'):
        matched = match_skills(resume_data['skills'], job_data['required_skills'])
        if job_data['required_skills']:  # Avoid division by zero
            score = len(matched) / len(job_data['required_skills']) * 100
    
    return score

if __name__ == "__main__":
    resume_file = "./SOURAV_NEW_RESUME.pdf"
    
    # Extract plain text
    resume_text = pdf_reader(resume_file)
    
    if resume_text:
        print("====== EXTRACTED TEXT ======")
        # print(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
        print(resume_text.encode("utf-8",errors="ignore"))
        
        resume_structured_data = extract_info_custom(resume_text)
        
        print("\n====== STRUCTURED RESUME DATA ======")
        for key, value in resume_structured_data.items():
            print(f"{key}: {value}")
            
        job_data = {
            "required_skills": ["Python", "Data Analysis", "Machine Learning"]
        }
        
        match_score = score_resume(resume_structured_data, job_data)
        print(f"\nResume match score: {match_score:.2f}%")
    else:
        print("Could not extract text from resume")