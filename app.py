import io
import os
import json
import argparse
from datetime import datetime
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import spacy
import re
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import Counter

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("You may need to install it with: python -m spacy download en_core_web_sm")
    exit(1)

def pdf_reader(file):
    """Extract text from PDF files"""
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

def extract_education(text):
    """Extract education information"""
    education_keywords = ['education', 'university', 'college', 'bachelor', 'master', 'phd', 'degree', 'diploma']
    education_info = []
    
    # Split text into sections
    lines = text.split('\n')
    in_education_section = False
    edu_section = ""
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Check if this line marks the start of education section
        if any(keyword in line_lower for keyword in education_keywords) and len(line_lower) < 30:
            in_education_section = True
            edu_section = line + "\n"
            
            # Look at next few lines to capture education details
            for j in range(1, 10):  # Look at next 10 lines max
                if i+j < len(lines):
                    edu_section += lines[i+j] + "\n"
            
            # Use regex to extract degree and institution
            degree_pattern = r'(?:bachelor|master|phd|b\.?[a-z]*|m\.?[a-z]*|ph\.?d)[\s\.]+(?:of|in)?\s+[a-z\s]+(?:engineering|science|arts|commerce|business|administration|technology)'
            institution_pattern = r'(?:university|college|institute|school) of [a-z\s]+|[a-z\s]+ (?:university|college|institute|school)'
            
            degrees = re.findall(degree_pattern, edu_section.lower())
            institutions = re.findall(institution_pattern, edu_section.lower())
            
            if degrees or institutions:
                education_info.append({
                    'degree': degrees[0].strip().title() if degrees else "",
                    'institution': institutions[0].strip().title() if institutions else ""
                })
            else:
                education_info.append({'raw': edu_section.strip()})
    
    return education_info

def extract_experience(text):
    """Extract work experience information"""
    experience_keywords = ['experience', 'work history', 'employment', 'career', 'job history']
    experience_info = []
    
    # Split text into sections
    lines = text.split('\n')
    in_exp_section = False
    exp_section = ""
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Check if this line marks the start of experience section
        if any(keyword in line_lower for keyword in experience_keywords) and len(line_lower) < 30:
            in_exp_section = True
            exp_section = line + "\n"
            
            # Look at next several lines to capture experience details
            for j in range(1, 20):  # Look at next 20 lines max
                if i+j < len(lines):
                    exp_section += lines[i+j] + "\n"
            
            # Extract company names, positions and dates
            company_pattern = r'(?:at|with)?\s([A-Z][A-Za-z0-9\'\-\&\.\s]{2,50})'
            position_pattern = r'([A-Z][a-z]+\s+[A-Za-z]+(?:\s+[A-Za-z]+)?(?:\s+[A-Za-z]+)?)\s+(?:at|in)'
            date_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]+\d{4}\s+(?:to|--|–)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]+\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]+\d{4}\s+(?:to|--|–)\s+(?:Present|Current|Now)'
            
            companies = re.findall(company_pattern, exp_section)
            positions = re.findall(position_pattern, exp_section)
            dates = re.findall(date_pattern, exp_section)
            
            # Add experience details
            if positions or companies or dates:
                experience_info.append({
                    'position': positions[0].strip() if positions else "",
                    'company': companies[0].strip() if companies else "",
                    'duration': dates[0].strip() if dates else ""
                })
            
            # Extract responsibilities using NLP
            doc = nlp(exp_section)
            responsibilities = []
            for sent in doc.sents:
                if len(sent.text.strip()) > 10 and ('develop' in sent.text.lower() or 
                                                   'manage' in sent.text.lower() or 
                                                   'create' in sent.text.lower() or
                                                   'lead' in sent.text.lower() or
                                                   'implement' in sent.text.lower()):
                    responsibilities.append(sent.text.strip())
            
            if responsibilities:
                if experience_info:
                    experience_info[-1]['responsibilities'] = responsibilities[:3]  # Limit to 3 key responsibilities
    
    return experience_info

def extract_info_custom(text):
    """Extract comprehensive information from resume text"""
    doc = nlp(text)
    
    # Initialize data dictionary
    data = {
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "education": [],
        "experience": [],
        "languages": [],
        "certifications": []
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
    
    # Extract GitHub/LinkedIn profiles
    github_pattern = r'github\.com\/[A-Za-z0-9_-]+'
    linkedin_pattern = r'linkedin\.com\/in\/[A-Za-z0-9_-]+'
    
    github = re.findall(github_pattern, text)
    linkedin = re.findall(linkedin_pattern, text)
    
    if github:
        data["github"] = github[0]
    if linkedin:
        data["linkedin"] = linkedin[0]
    
    # Expanded skill keywords
    skill_keywords = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "html", "css", "php", "ruby", "c++", "c#", "swift", "kotlin", "go", "rust", "scala", 
        # Frameworks & Libraries
        "react", "angular", "vue", "node", "django", "flask", "spring", "laravel", "express", "fastapi", "tensorflow", "pytorch", "pandas", "numpy",
        # Databases
        "sql", "mysql", "postgresql", "mongodb", "firebase", "oracle", "redis", "cassandra", "dynamodb", "sqlite",  
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ansible", "git", "github", "cicd", "devops",
        # Data Science & Analytics
        "machine learning", "data analysis", "data science", "big data", "data visualization", "statistics", "nlp", "computer vision", "ai", 
        # Business Intelligence
        "excel", "powerbi", "tableau", "looker", "dax", "power query", "r", "spss", "sas", 
        # Project Management
        "agile", "scrum", "kanban", "jira", "trello", "project management", "waterfall", "pmp", "prince2",
        # Soft Skills
        "leadership", "teamwork", "communication", "problem solving", "critical thinking", "time management",
        # Marketing
        "seo", "sem", "google analytics", "social media", "content marketing", "email marketing",
        # Design
        "photoshop", "illustrator", "figma", "sketch", "ui design", "ux design", "adobe creative suite"
    ]
    
    found_skills = []
    text_lower = text.lower()
    for skill in skill_keywords:
        if skill in text_lower:
            found_skills.append(skill.title())  # Capitalize for consistency
    
    data["skills"] = found_skills
    
    # Extract name from first few lines
    lines = text.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if line and len(line) < 40 and not any(c in line for c in "@.,:;()[]{}"):
            data["name"] = line
            break
    
    # Extract education information
    data["education"] = extract_education(text)
    
    # Extract experience information
    data["experience"] = extract_experience(text)
    
    # Extract languages
    language_section = re.search(r'languages?:?.*?(?:\n|$)((?:.*?\n?){1,5})', text.lower())
    if language_section:
        lang_text = language_section.group(1)
        # Common languages
        languages = ["english", "spanish", "french", "german", "chinese", "japanese", "russian", 
                     "arabic", "hindi", "portuguese", "italian", "dutch", "korean", "swedish", "turkish"]
        for lang in languages:
            if lang in lang_text.lower():
                data["languages"].append(lang.title())
    
    # Extract certifications
    cert_patterns = [
        r'certified\s+[a-z\s]+(?:developer|engineer|administrator|architect|specialist|professional)',
        r'[a-z]+\s+certification',
        r'[A-Z]{2,}(?:-[A-Z\d]+)+' # For abbreviations like AWS-SAA, MCSA, etc.
    ]
    
    certs = []
    for pattern in cert_patterns:
        found = re.findall(pattern, text.lower())
        certs.extend([c.strip().title() for c in found])
    
    if certs:
        data["certifications"] = list(set(certs))  # Remove duplicates
    
    return data

def match_skills(resume_skills, job_requirement):
    """Match resume skills against job requirements"""
    resume_skills_lower = [skill.lower() for skill in resume_skills]
    job_requirement_lower = [skill.lower() for skill in job_requirement]
    
    matched_skills = []
    unmatched_skills = []
    
    for skill in job_requirement_lower:
        found = False
        for resume_skill in resume_skills_lower:
            if skill in resume_skill or resume_skill in skill:
                matched_skills.append(skill)
                found = True
                break
        if not found:
            unmatched_skills.append(skill)
    
    return matched_skills, unmatched_skills

def score_resume(resume_data, job_data):
    """Score the resume based on job data"""
    if not resume_data or not job_data:
        return 0, [], []
    
    score = 0
    matched_skills = []
    missing_skills = []
    
    if resume_data.get('skills') and job_data.get('required_skills'):
        matched, unmatched = match_skills(resume_data['skills'], job_data['required_skills'])
        matched_skills = matched
        missing_skills = unmatched
        
        if job_data['required_skills']:  # Avoid division by zero
            score = len(matched) / len(job_data['required_skills']) * 100
    
    return score, matched_skills, missing_skills

def generate_suggestions(resume_data, job_data, score, missing_skills):
    """Generate personalized suggestions based on resume analysis"""
    suggestions = []
    
    # Suggestion based on skills match
    if score < 50:
        suggestions.append(f"Your skills match is below 50%. Consider upskilling in: {', '.join(missing_skills)}")
    elif score < 75:
        suggestions.append(f"To improve your chances, consider gaining more experience in: {', '.join(missing_skills)}")
    else:
        suggestions.append("Your skills match the job requirements well!")
    
    # Check for contact information
    if not resume_data.get('email') or not resume_data.get('phone'):
        suggestions.append("Add complete contact information (email and phone) to your resume")
    
    # Check for LinkedIn profile
    if not resume_data.get('linkedin'):
        suggestions.append("Consider adding your LinkedIn profile URL to increase credibility")
    
    # Check experience descriptions
    if resume_data.get('experience'):
        has_metrics = False
        for exp in resume_data['experience']:
            if 'responsibilities' in exp:
                for resp in exp.get('responsibilities', []):
                    if any(metric in resp.lower() for metric in ['increase', 'reduce', 'improve', '%', 'percent', 'decreased', 'grew']):
                        has_metrics = True
                        break
        
        if not has_metrics:
            suggestions.append("Add measurable achievements with metrics to your experience descriptions")
    
    # Check for relevant certifications
    if job_data.get('preferred_certifications') and resume_data.get('certifications'):
        cert_matches = [c for c in job_data['preferred_certifications'] if any(rc.lower() in c.lower() or c.lower() in rc.lower() for rc in resume_data['certifications'])]
        if not cert_matches and job_data['preferred_certifications']:
            suggestions.append(f"Consider obtaining relevant certifications like: {', '.join(job_data['preferred_certifications'])}")
    
    # Check resume length indirectly
    word_count = len(resume_data.get('skills', [])) + len(resume_data.get('education', [])) + len(resume_data.get('experience', []))
    if word_count < 10:
        suggestions.append("Your resume may be too brief. Consider adding more details about your experience and skills")
    
    return suggestions

def visualize_skills_match(matched, missing, filename):
    """Create a visualization of skills match"""
    labels = ['Matched Skills', 'Missing Skills']
    sizes = [len(matched), len(missing)]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # explode 1st slice
    
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  
    plt.title('Skills Match Analysis')
    
    # Add legend with the actual skills
    matched_legend = '\n'.join([f"✓ {skill.title()}" for skill in matched])
    missing_legend = '\n'.join([f"✗ {skill.title()}" for skill in missing])
    
    plt.figtext(0.9, 0.5, f"Matched Skills:\n{matched_legend}", wrap=True, horizontalalignment='center')
    plt.figtext(0.1, 0.5, f"Missing Skills:\n{missing_legend}", wrap=True, horizontalalignment='center')
    
    plt.savefig(filename)
    plt.close()
    
    return filename

def generate_report(resume_data, job_data, score, matched_skills, missing_skills, suggestions, suitable_jobs=None):
    """Generate a comprehensive report"""
    report = {
        "candidate_name": resume_data.get("name", "Unknown"),
        "contact": {
            "email": resume_data.get("email", ""),
            "phone": resume_data.get("phone", "")
        },
        "match_score": score,
        "skills_analysis": {
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "total_required_skills": len(job_data.get("required_skills", [])),
            "total_matched_skills": len(matched_skills)
        },
        "education": resume_data.get("education", []),
        "experience": resume_data.get("experience", []),
        "suggestions": suggestions,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add suitable jobs if provided
    if suitable_jobs:
        report["suitable_job_recommendations"] = suitable_jobs
    
    return report

def save_report_as_json(report, filename):
    """Save report as JSON file"""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=4)
    return filename

def print_report(report):
    """Print formatted report to console"""
    print("\n====== RESUME ANALYSIS REPORT ======")
    print(f"Candidate: {report['candidate_name']}")
    print(f"Contact: {report['contact']['email']} | {report['contact']['phone']}")
    print(f"\nMatch Score: {report['match_score']:.2f}%")
    
    print("\nSkills Analysis:")
    print(f"- Matched Skills ({len(report['skills_analysis']['matched_skills'])}): {', '.join(report['skills_analysis']['matched_skills'])}")
    print(f"- Missing Skills ({len(report['skills_analysis']['missing_skills'])}): {', '.join(report['skills_analysis']['missing_skills'])}")
    
    print("\nSuggestions:")
    for i, suggestion in enumerate(report['suggestions'], 1):
        print(f"{i}. {suggestion}")
    
    # Print suitable job recommendations if available
    if 'suitable_job_recommendations' in report:
        print("\nRecommended Job Roles:")
        for i, job in enumerate(report['suitable_job_recommendations'], 1):
            print(f"{i}. {job['title']} (Match: {job['match_score']:.2f}%)")
            print(f"   Key skills: {', '.join(job['key_skills'])}")
    
    print(f"\nAnalysis Date: {report['analysis_date']}")
    print("=====================================")

def analyze_resumes(job_data, resume_files, job_database=None):
    """Analyze multiple resumes against job requirements"""
    results = []
    
    for resume_file in resume_files:
        print(f"\nAnalyzing resume: {resume_file}...")
        resume_text = pdf_reader(resume_file)
        
        if resume_text:
            resume_data = extract_info_custom(resume_text)
            
            score, matched_skills, missing_skills = score_resume(resume_data, job_data)
            suggestions = generate_suggestions(resume_data, job_data, score, missing_skills)
            
            # Find suitable jobs if job database is provided
            suitable_jobs = None
            if job_database:
                suitable_jobs = find_suitable_jobs(resume_data, job_database)
            
            # Generate visualization
            chart_filename = f"skills_match_{os.path.basename(resume_file).split('.')[0]}.png"
            visualize_skills_match(matched_skills, missing_skills, chart_filename)
            
            # Generate and save report
            report = generate_report(resume_data, job_data, score, matched_skills, missing_skills, suggestions, suitable_jobs)
            report_filename = f"report_{os.path.basename(resume_file).split('.')[0]}.json"
            save_report_as_json(report, report_filename)
            
            # Print report
            print_report(report)
            
            results.append({
                "filename": resume_file,
                "name": resume_data.get("name", "Unknown"),
                "score": score,
                "matched_skills": len(matched_skills),
                "missing_skills": len(missing_skills),
                "report_file": report_filename,
                "chart_file": chart_filename
            })
        else:
            print(f"Could not extract text from resume: {resume_file}")
    
    return results

def rank_candidates(results):
    """Rank candidates based on their scores"""
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    
    print("\n====== CANDIDATE RANKING ======")
    headers = ["Rank", "Name", "Match Score", "Matched Skills", "Missing Skills"]
    table_data = []
    
    for i, result in enumerate(ranked, 1):
        table_data.append([
            i, 
            result["name"], 
            f"{result['score']:.2f}%", 
            result["matched_skills"],
            result["missing_skills"]
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    return ranked

def find_suitable_jobs(resume_data, job_database):
    """Find suitable jobs for a candidate based on their skills and experience"""
    suitable_jobs = []
    
    if not resume_data.get('skills'):
        return suitable_jobs
    
    for job in job_database:
        # Calculate skill match
        matched, _ = match_skills(resume_data['skills'], job['required_skills'])
        match_score = len(matched) / len(job['required_skills']) * 100 if job['required_skills'] else 0
        
        # Consider job suitable if match score is above 50%
        if match_score >= 50:
            suitable_jobs.append({
                'title': job['title'],
                'match_score': match_score,
                'key_skills': matched,
                'description': job.get('description', '')
            })
    
    # Sort by match score (highest first)
    suitable_jobs = sorted(suitable_jobs, key=lambda x: x['match_score'], reverse=True)
    
    return suitable_jobs[:5]  # Return top 5 suitable jobs

def load_job_database(file_path):
    """Load job database from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Job database file {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not parse job database file {file_path}.")
        return []

def analyze_job_description(job_text):
    """Extract skills, experience, and certifications from job description text"""
    doc = nlp(job_text)
    
    # Initialize job data
    job_data = {
        "title": "",
        "required_skills": [],
        "experience_required": "",
        "preferred_certifications": [],
        "education_required": "",
        "description": job_text[:500] + "..." if len(job_text) > 500 else job_text  # Store truncated description
    }
    
    # Extract job title
    lines = job_text.split('\n')
    for line in lines[:5]:  # Check first 5 lines for job title
        line = line.strip()
        if line and len(line) < 60 and not line.lower().startswith(('job', 'position')):
            job_data["title"] = line
            break
    
    # Known skill keywords to look for
    skill_keywords = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "html", "css", "php", "ruby", "c++", "c#", "swift", "kotlin", "go", "rust", "scala", 
        # Frameworks & Libraries
        "react", "angular", "vue", "node", "django", "flask", "spring", "laravel", "express", "fastapi", "tensorflow", "pytorch", "pandas", "numpy",
        # Databases
        "sql", "mysql", "postgresql", "mongodb", "firebase", "oracle", "redis", "cassandra", "dynamodb", "sqlite",  
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ansible", "git", "github", "cicd", "devops",
        # Data Science & Analytics
        "machine learning", "data analysis", "data science", "big data", "data visualization", "statistics", "nlp", "computer vision", "ai", 
        # Business Intelligence
        "excel", "powerbi", "tableau", "looker", "dax", "power query", "r", "spss", "sas", 
        # Project Management
        "agile", "scrum", "kanban", "jira", "trello", "project management", "waterfall", "pmp", "prince2",
        # Soft Skills
        "leadership", "teamwork", "communication", "problem solving", "critical thinking", "time management",
        # Marketing
        "seo", "sem", "google analytics", "social media", "content marketing", "email marketing",
        # Design
        "photoshop", "illustrator", "figma", "sketch", "ui design", "ux design", "adobe creative suite"
    ]
    
    # Extract skills
    found_skills = []
    job_text_lower = job_text.lower()
    for skill in skill_keywords:
        if skill in job_text_lower:
            found_skills.append(skill.title())  # Capitalize for consistency
    
    job_data["required_skills"] = found_skills
    
    # Extract experience requirements
    experience_patterns = [
        r'(\d+)[\+\-]?\s+years?\s+(?:of\s+)?experience',
        r'experience\s+(?:of|for)?\s+(\d+)[\+\-]?\s+years?',
        r'minimum\s+(?:of\s+)?(\d+)[\+\-]?\s+years?\s+(?:of\s+)?experience'
    ]
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, job_text_lower)
        if matches:
            job_data["experience_required"] = f"{matches[0]}+ years"
            break
    
    # Extract certification requirements
    cert_patterns = [
        r'(?:certification|certified)\s+(?:in|as)\s+([a-z\s]+(?:developer|engineer|administrator|architect|specialist|professional))',
        r'([a-z]+\s+certification)',
        r'([A-Z]{2,}(?:-[A-Z\d]+)+)' # For abbreviations like AWS-SAA, MCSA, etc.
    ]
    
    certs = []
    for pattern in cert_patterns:
        found = re.findall(pattern, job_text_lower)
        certs.extend([c.strip().title() for c in found])
    
    if certs:
        job_data["preferred_certifications"] = list(set(certs))  # Remove duplicates
    
    # Extract education requirements
    education_patterns = [
        r'(?:bachelor|master|phd|b\.?[a-z]*|m\.?[a-z]*|ph\.?d)[\s\.]+(?:of|in)?\s+[a-z\s]+(?:engineering|science|arts|commerce|business|administration|technology)',
        r'(?:bachelor|master|phd|graduate|undergraduate)\s+degree',
        r'degree\s+in\s+[a-z\s]+'
    ]
    
    for pattern in education_patterns:
        matches = re.findall(pattern, job_text_lower)
        if matches:
            job_data["education_required"] = matches[0].strip().title()
            break
    
    return job_data

def save_job_to_database(job_data, database_file):
    """Save a job to the job database"""
    job_database = []
    
    # Load existing database if it exists
    try:
        with open(database_file, 'r') as f:
            job_database = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        job_database = []
    
    # Add new job if it doesn't already exist
    job_exists = False
    for job in job_database:
        if job.get('title') == job_data.get('title'):
            job_exists = True
            break
    
    if not job_exists:
        job_database.append(job_data)
        
        # Save updated database
        with open(database_file, 'w') as f:
            json.dump(job_database, f, indent=4)
        
        print(f"Job '{job_data.get('title')}' added to database.")
    else:
        print(f"Job '{job_data.get('title')}' already exists in database.")
    
    return job_data

def main():
    """Main function to run the resume analyzer"""
    parser = argparse.ArgumentParser(description='Enhanced Resume Analyzer')
    parser.add_argument('--resumes', nargs='+', help='Paths to resume PDF files')
    parser.add_argument('--skills', nargs='+', help='Required skills for the job')
    parser.add_argument('--certifications', nargs='+', help='Preferred certifications for the job')
    parser.add_argument('--output-dir', default=".", help='Output directory for reports and charts')
    parser.add_argument('--summary', action='store_true', help='Generate a summary report for all candidates')
    parser.add_argument('--job-database', default="job_database.json", help='Path to job database file')
    parser.add_argument('--analyze-job', action='store_true', help='Analyze a job description')
    parser.add_argument('--job-description-file', help='Path to job description file')
    
    args = parser.parse_args()
    
    # Change to output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    # Handle job description analysis
    if args.analyze_job:
        if not args.job_description_file:
            print("Error: Please provide a job description file with --job-description-file")
            return
        
        job_text = pdf_reader(args.job_description_file)
        if job_text:
            job_data = analyze_job_description(job_text)
            save_job_to_database(job_data, args.job_database)
            print(f"Job description analyzed and saved to {args.job_database}")
        else:
            print("Error: Could not read job description file")
            return
    
    # Process resumes
    if args.resumes:
        # Create job data from arguments or load from database
        job_data = {}
        if args.skills:
            job_data["required_skills"] = args.skills
        if args.certifications:
            job_data["preferred_certifications"] = args.certifications
        
        # If job data is empty, use the last job from database
        if not job_data.get("required_skills"):
            job_database = load_job_database(args.job_database)
            if job_database:
                job_data = job_database[-1]  # Use the most recently added job
                print(f"Using job '{job_data.get('title', 'Unknown')}' from database for analysis")
            else:
                print("Error: No job data provided. Please specify skills or add a job to the database")
                return
        
        # Analyze resumes
        results = analyze_resumes(job_data, args.resumes, load_job_database(args.job_database))
        
        # Generate summary if requested
        if args.summary and results:
            rank_candidates(results)
    else:
        if not args.analyze_job:
            print("Error: Please provide resume files with --resumes or use --analyze-job to analyze a job description")
            return

if __name__ == "__main__":
    main()