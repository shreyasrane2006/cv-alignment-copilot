"""
Pydantic schemas for structured CV / JD extraction.
These define the exact JSON shape the LLM must return — Instructor
enforces this schema against Ollama's structured-output support, so
the parser can't return malformed or missing fields.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class EducationEntry(BaseModel):
    degree: str = Field(description="Degree name/title, e.g. 'MSc Advanced Computer Science'")
    institution: str = Field(description="Name of the university/college")
    dates: Optional[str] = Field(default=None, description="Date range as written in the CV, if present")


class ExperienceEntry(BaseModel):
    role: str = Field(description="Job title")
    company: str = Field(description="Employer name")
    dates: Optional[str] = Field(default=None, description="Date range as written in the CV, if present")
    description: List[str] = Field(
        default_factory=list,
        description="Bullet points describing responsibilities/achievements, each as its own string"
    )


class ProjectEntry(BaseModel):
    name: str = Field(description="Project title")
    description: str = Field(description="Short description of what was built/done")


class CVData(BaseModel):
    personal_summary: Optional[str] = Field(
        default=None, description="The professional summary/objective paragraph, if present"
    )
    education: List[EducationEntry] = Field(default_factory=list)
    skills: List[str] = Field(
        default_factory=list,
        description="Flat list of individual skills/technologies/languages/tools mentioned anywhere in the CV"
    )
    experience: List[ExperienceEntry] = Field(default_factory=list)
    projects: List[ProjectEntry] = Field(default_factory=list)
    certifications: List[str] = Field(
        default_factory=list,
        description="Named certifications/courses completed (e.g. 'AWS Certified Cloud Practitioner'). Leave empty if none are stated."
    )
    links: List[str] = Field(
        default_factory=list,
        description="URLs explicitly present in the CV (GitHub, LinkedIn, portfolio, published papers, etc.). Leave empty if none are stated."
    )
    achievements: List[str] = Field(
        default_factory=list,
        description="Awards, honors, or notable recognitions explicitly stated. Leave empty if none are stated."
    )


class JDData(BaseModel):
    job_title: Optional[str] = Field(default=None)
    required_skills: List[str] = Field(
        default_factory=list,
        description="Skills/technologies explicitly stated as required/must-have"
    )
    preferred_skills: List[str] = Field(
        default_factory=list,
        description="Skills/technologies explicitly stated as preferred/desirable/nice-to-have"
    )
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Key responsibilities/duties of the role, each as its own string"
    )
    qualifications: List[str] = Field(
        default_factory=list,
        description="Non-skill requirements such as years of experience, degree level, or certifications required (e.g. '2+ years experience', 'Master's degree'). Keep these separate from required_skills — a qualification is a credential/tenure requirement, not a technical skill."
    )