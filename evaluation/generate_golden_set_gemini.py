"""
generate_golden_set_gemini.py

Bypasses the local PyMuPDF parser and uploads scanned PDFs directly to Gemini 2.5 Flash.
Includes automatic Rate Limit (429) handling to respect the Free Tier API limits.
"""

import os
import json
import time
from pydantic import BaseModel, Field
from google import genai

# 1. Initialize the Gemini Client
try:
    gemini_client = genai.Client()
except Exception as e:
    print("Failed to initialize Gemini Client. Did you set the GEMINI_API_KEY environment variable?")
    exit(1)

GEMINI_MODEL = "gemini-3.5-flash-lite"

# 2. Define the Strict Pydantic Schema for the Output
class EvaluationPair(BaseModel):
    requirement: str = Field(description="A realistic job requirement for this candidate's field.")
    cv_evidence: str = Field(description="The exact snippet/bullet point from the CV used to make the judgment. Use 'None' if it's a No Match.")
    gold_label: str = Field(description="Must be exactly one of: Match, Partial Match, No Match")
    rationale: str = Field(description="A 1-sentence logical justification for this label.")

class GoldenSetBatch(BaseModel):
    pairs: list[EvaluationPair] = Field(description="Exactly 5 evaluation pairs")

# 3. The Generation Prompt
GENERATION_PROMPT = """You are an expert technical recruiter and AI evaluator.
I have attached a candidate's CV as a PDF document. It is a scanned image, so please read the visual text carefully.
Your task is to extract their background and generate EXACTLY 5 job requirement evaluation pairs to test an AI scoring system.

You MUST provide this exact distribution for the 5 pairs:
- 2 'Match' pairs: The requirement is explicitly proven by the CV evidence.
- 2 'Partial Match' pairs: The requirement is adjacent, vaguely supported, or uses a different but related tool.
- 1 'No Match' pair: The requirement is completely absent from the CV.

Make the requirements realistic to the candidate's specific industry.
"""

def generate_questions_from_pdf(filepath: str, filename: str) -> list[dict]:
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"   -> Uploading {filename} to Gemini for native OCR and generation...")
            uploaded_file = gemini_client.files.upload(file=filepath)
            
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    uploaded_file,
                    GENERATION_PROMPT
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GoldenSetBatch,
                    "temperature": 0.2,
                }
            )
            
            # Immediately delete the file from Google's servers
            gemini_client.files.delete(name=uploaded_file.name)
            
            generated_batch = response.parsed
            
            output = []
            if generated_batch and generated_batch.pairs:
                for pair in generated_batch.pairs:
                    pair_dict = pair.model_dump()
                    pair_dict["source_file"] = filename
                    output.append(pair_dict)
                
            return output
            
        except Exception as e:
            error_msg = str(e)
            # Check if the error is a Rate Limit (429)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                print(f"   -> [Rate Limit Hit] Waiting 60 seconds before retrying (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(60)
            else:
                print(f"   -> API Error during generation: {e}")
                return []
                
    print(f"   -> Failed to process {filename} after {max_retries} attempts.")
    return []

def main():
    resume_dir = "./Resume"
    output_file = "golden_dataset_gemini.json"
    master_dataset = []

    if not os.path.exists(resume_dir):
        print(f"Error: Directory '{resume_dir}' not found.")
        return

    pdf_files = [f for f in os.listdir(resume_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} resumes. Commencing rate-limit-aware generation...\n")

    for idx, filename in enumerate(pdf_files, 1):
        filepath = os.path.join(resume_dir, filename)
        print(f"[{idx}/{len(pdf_files)}] Processing {filename}...")
        
        new_pairs = generate_questions_from_pdf(filepath, filename)
        
        if not new_pairs:
            continue
            
        master_dataset.extend(new_pairs)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(master_dataset, f, indent=4)
            
        print(f"   -> Success! Dataset now has {len(master_dataset)} pairs.")
        
        # Increased base delay to 5 seconds to throttle requests naturally
        time.sleep(10)

    print(f"\nGeneration complete! {len(master_dataset)} total questions saved to {output_file}.")

if __name__ == "__main__":
    main()