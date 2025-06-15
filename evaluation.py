"""
Fact-Based Answer Evaluator - FIXED VERSION
===========================================

A comprehensive evaluation system that uses LLM-based fact decomposition
to objectively compare AgentEval outputs against ground truth responses.

FIXED: 
- Generates ultra-short questions (3-4 word answers max)
- Saves results to ./data/evaluation/ directory
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

@dataclass
class EvaluationResult:
    """Data class for storing evaluation results"""
    question: str
    gt_answer: str
    llm_answer: str
    match: bool
    confidence: str
    reasoning: str

@dataclass
class FactEvaluationSummary:
    """Summary of the complete fact-based evaluation"""
    total_questions: int
    correct_facts: int
    accuracy_percentage: float
    evaluation_results: List[EvaluationResult]
    timestamp: str

class GeminiClient:
    """Wrapper for Google Gemini API client"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.1):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        
    def generate_response(self, prompt: str) -> str:
        """Generate response using Gemini"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=2048,
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

class ObjectiveQuestionGenerator:
    """Generates ultra-short, objective questions with 3-4 word answers"""
    
    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
        
    def generate_questions(self, topic_definition: str, scoring_rubric: str = "") -> List[str]:
        """Generate ultra-short questions that have very brief factual answers"""
        
        prompt = f"""
You are an Ultra-Precise Question Generator for Corporate Governance Evaluation.

CRITICAL REQUIREMENT: Generate questions that can be answered with MAXIMUM 3-4 WORDS.


VALID ANSWER TYPES (3-4 words max):
- Dates: "March 31, 2024"
- Numbers: "11%", "85 days", "7 directors"
- Names: "John Smith", "ABC Company"
- Yes/No: "Yes", "No"
- Classifications: "Permanent", "Non-permanent", "Independent"
- Scores: "Score 2", "0 out of 2"
- Counts: "5 members", "3 documents"
- Pages: "Page 15", "pp. 10-12"

INVALID QUESTIONS (answers too long):
❌ "What is the detailed explanation of board independence?"
❌ "How does the company ensure diversity?"
❌ "What are the comprehensive governance policies?"

VALID QUESTIONS (short answers):
✅ "Financial year end date?"
✅ "AGM date?"
✅ "Women workforce percentage?"
✅ "Total board members?"
✅ "Score assigned?"

TOPIC DEFINITION:
{topic_definition}

SCORING RUBRIC:
{scoring_rubric}

FORMAT: Just list the questions, one per line, no numbering.
"""
        
        response = self.client.generate_response(prompt)
        
        # Parse questions from response and clean them up
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('FORMAT'):
                # Remove numbering and formatting
                question = line
                # Remove leading numbers, bullets, etc.
                question = question.lstrip('0123456789.- •').strip()
                # Remove brackets if present
                question = question.replace('[', '').replace(']', '')
                
                if question and len(question) > 5:  # Valid question
                    questions.append(question)
        
        # Ensure we have exactly 8 questions, pad if necessary
        if len(questions) < 8:
            # Add standard questions if we don't have enough
            standard_questions = [
                "Final score assigned?",
                "Key ratio mentioned?",
                "Key date mentioned?",
                "Main percentage stated?",
                "Total count given?",
                "Compliance status?",
                "Key person mentioned?",
                "Calculation result?"
            ]
            
            for q in standard_questions:
                if q not in questions and len(questions) < 8:
                    questions.append(q)
        
        return questions[:8]  # Return exactly 8 questions

class FactExtractor:
    """Extracts ultra-short factual answers (3-4 words max)"""
    
    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
        
    def extract_fact(self, text: str, question: str) -> str:
        """Extract ultra-short factual answer (3-4 words maximum)"""
        
        prompt = f"""
You are an Ultra-Precise Fact Extractor.

CRITICAL: Your answer must be MAXIMUM 3-4 WORDS. No exceptions.

EXTRACTION RULES:
1. Maximum 3-4 words only
2. Extract the most specific fact
3. Use exact numbers, dates, percentages from text
4. If not found, answer "Not specified"
5. For multiple items, pick the most relevant one

EXAMPLES:
Question: "Financial year end date?" → Answer: "March 31, 2024"
Question: "Women workforce percentage?" → Answer: "11%"
Question: "Total board members?" → Answer: "7 directors"
Question: "Final score assigned?" → Answer: "Score 2"
Question: "Primary source page?" → Answer: "Page 15"
Question: "AGM date?" → Answer: "June 24, 2024"

TEXT:
{text}

QUESTION:
{question}

ULTRA-SHORT ANSWER (3-4 words max):
"""
        
        answer = self.client.generate_response(prompt)
        
        # Ensure answer is truly short
        words = answer.strip().split()
        if len(words) > 4:
            # Truncate to first 4 words if too long
            answer = ' '.join(words[:4])
        
        return answer.strip()

class FactComparator:
    """Compares ultra-short facts between ground truth and LLM answers"""
    
    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
        
    def compare_facts(self, question: str, gt_answer: str, llm_answer: str) -> Dict[str, Any]:
        """Compare two ultra-short factual answers"""
        
        prompt = f"""
You are comparing two ultra-short factual answers (3-4 words each).

COMPARISON RULES:
1. Exact matches = MATCH
2. Same date in different format = MATCH (e.g., "March 31, 2024" vs "31 March 2024")
3. Same percentage = MATCH (e.g., "11%" vs "11 percent")
4. Same number = MATCH (e.g., "7 directors" vs "7 members")
5. Synonymous terms = MATCH (e.g., "Score 2" vs "2 points")
6. Any factual difference = NO MATCH
7. "Not specified" vs actual fact = NO MATCH

QUESTION: {question}
GROUND TRUTH: {gt_answer}
LLM ANSWER: {llm_answer}

Respond in JSON format:
{{
    "match": true/false,
    "confidence": "high/medium/low",
    "reasoning": "Brief reason",
    "discrepancies": []
}}
"""
        
        response = self.client.generate_response(prompt)
        
        # Parse JSON response
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "match": result.get("match", False),
                    "confidence": result.get("confidence", "low"),
                    "reasoning": result.get("reasoning", ""),
                    "discrepancies": result.get("discrepancies", [])
                }
        except (json.JSONDecodeError, AttributeError):
            print(f"Failed to parse comparison result: {response}")
        
        # Simple fallback comparison for ultra-short answers
        gt_clean = gt_answer.lower().strip()
        llm_clean = llm_answer.lower().strip()
        
        # Direct match
        if gt_clean == llm_clean:
            return {"match": True, "confidence": "high", "reasoning": "Exact match", "discrepancies": []}
        
        # Close match (handle common variations)
        if self._are_equivalent(gt_clean, llm_clean):
            return {"match": True, "confidence": "medium", "reasoning": "Equivalent values", "discrepancies": []}
        
        return {"match": False, "confidence": "high", "reasoning": "Different facts", "discrepancies": [f"{gt_answer} != {llm_answer}"]}
    
    def _are_equivalent(self, answer1: str, answer2: str) -> bool:
        """Check if two short answers are equivalent"""
        # Remove common variations
        variations = [
            ("score ", ""), ("page ", ""), ("pp. ", ""), ("%", " percent"),
            (",", ""), ("directors", "members"), ("march", "mar"),
            ("january", "jan"), ("february", "feb"), ("april", "apr"),
            ("june", "jun"), ("july", "jul"), ("august", "aug"),
            ("september", "sep"), ("october", "oct"), ("november", "nov"),
            ("december", "dec")
        ]
        
        clean1, clean2 = answer1, answer2
        for old, new in variations:
            clean1 = clean1.replace(old, new)
            clean2 = clean2.replace(old, new)
        
        # Check if they're now the same
        return clean1.strip() == clean2.strip()

class FactBasedEvaluator:
    """Main evaluator for ultra-short fact-based evaluation"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.1):
        self.gemini_client = GeminiClient(model_name, temperature)
        self.question_generator = ObjectiveQuestionGenerator(self.gemini_client)
        self.fact_extractor = FactExtractor(self.gemini_client)
        self.fact_comparator = FactComparator(self.gemini_client)
        
        # FIXED: Create evaluation directory
        self.eval_dir = "./data/evaluation/"
        os.makedirs(self.eval_dir, exist_ok=True)
        
    def evaluate(
        self, 
        topic_definition: str, 
        scoring_rubric: str, 
        ground_truth: str, 
        llm_output: str
    ) -> FactEvaluationSummary:
        """
        Perform complete ultra-short fact-based evaluation
        """
        
        print("🔍 Generating ultra-short questions (3-4 word answers)...")
        questions = self.question_generator.generate_questions(
            topic_definition, scoring_rubric
        )
        
        print(f"📋 Generated {len(questions)} ultra-short questions")
        for i, q in enumerate(questions, 1):
            print(f"   {i}. {q}")
        
        evaluation_results = []
        
        for i, question in enumerate(questions, 1):
            print(f"⚡ Processing question {i}/{len(questions)}: {question}")
            
            time.sleep(10)  # Simulate processing time
            
            # Extract ultra-short facts from both texts
            gt_answer = self.fact_extractor.extract_fact(ground_truth, question)
            llm_answer = self.fact_extractor.extract_fact(llm_output, question)
            
            print(f"   GT: '{gt_answer}' | LLM: '{llm_answer}'")
            
            # Compare the facts
            comparison_result = self.fact_comparator.compare_facts(
                question, gt_answer, llm_answer
            )
            
            match_icon = "✅" if comparison_result["match"] else "❌"
            print(f"   {match_icon} Match: {comparison_result['match']} ({comparison_result['confidence']})")
            
            evaluation_results.append(EvaluationResult(
                question=question,
                gt_answer=gt_answer,
                llm_answer=llm_answer,
                match=comparison_result["match"],
                confidence=comparison_result["confidence"],
                reasoning=comparison_result["reasoning"]
            ))
        
        # Calculate final accuracy
        correct_facts = sum(1 for result in evaluation_results if result.match)
        accuracy_percentage = (correct_facts / len(evaluation_results)) * 100
        
        print(f"\n📊 EVALUATION COMPLETE: {correct_facts}/{len(evaluation_results)} correct ({accuracy_percentage:.1f}%)")
        
        return FactEvaluationSummary(
            total_questions=len(evaluation_results),
            correct_facts=correct_facts,
            accuracy_percentage=accuracy_percentage,
            evaluation_results=evaluation_results,
            timestamp=datetime.now().isoformat()
        )
    
    def save_results(self, summary: FactEvaluationSummary, filename: str = None):
        """FIXED: Save evaluation results to ./data/evaluation/ directory"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fact_evaluation_{timestamp}"
        
        # FIXED: Use evaluation directory
        csv_path = os.path.join(self.eval_dir, f"{filename}.csv")
        json_path = os.path.join(self.eval_dir, f"{filename}.json")
        
        # Prepare data for CSV
        csv_data = []
        for i, result in enumerate(summary.evaluation_results, 1):
            csv_data.append({
                "Question_ID": i,
                "Question": result.question,
                "Ground_Truth": result.gt_answer,
                "LLM_Answer": result.llm_answer,
                "Match": "✅" if result.match else "❌",
                "Confidence": result.confidence,
                "Reasoning": result.reasoning
            })
        
        # Add summary row
        csv_data.append({
            "Question_ID": "SUMMARY",
            "Question": "=== FINAL RESULTS ===",
            "Ground_Truth": f"{summary.correct_facts}/{summary.total_questions}",
            "LLM_Answer": f"{summary.accuracy_percentage:.1f}% accurate",
            "Match": "✅ PASS" if summary.accuracy_percentage >= 70 else "❌ FAIL",
            "Confidence": "Overall",
            "Reasoning": f"Ultra-short fact accuracy: {summary.accuracy_percentage:.1f}%"
        })
        
        # Save CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        # Save detailed JSON
        json_data = {
            "evaluation_type": "ultra_short_facts",
            "summary": {
                "total_questions": summary.total_questions,
                "correct_facts": summary.correct_facts,
                "accuracy_percentage": summary.accuracy_percentage,
                "timestamp": summary.timestamp,
                "pass_threshold": 70.0,
                "result": "PASS" if summary.accuracy_percentage >= 70 else "FAIL"
            },
            "detailed_results": [
                {
                    "question_id": i,
                    "question": result.question,
                    "gt_answer": result.gt_answer,
                    "llm_answer": result.llm_answer,
                    "match": result.match,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning
                }
                for i, result in enumerate(summary.evaluation_results, 1)
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"📊 Results saved to:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")
        
        return csv_path, json_path

def main():
    """Example usage of the ultra-short fact-based evaluator"""
    
    # Example data
    topic_definition = """
     Question: POSH policy compliance
    
    Topic: POSH policy compliance
    Goal: To assess if the company has a proper POSH (Prevention of Sexual Harassment) policy and compliance
    Guidance: You have to check two main things. First if the company has POSH (Prevention of Sexual Harassment) policy. You can check in either annual report or in document having all policies. Second, check if the company has reported any complaints or cases under this policy in the last financial year. This information is typically found in the corporate governance report or the annual report. Make sure you quote this source in the answer with the page number from which you extract the information.
    """
    
    scoring_rubric = """
        Score 0: "if company does not have any policy regarding prevention of sexual harrassment and the company also has not provided information on the number of sexual harassment incidents.",
        Score 1: "if company has either policy regarding prevention of sexual harrassment or the company has provided information on the number of sexual harassment incidents.",
        Score 2: "if company has both policy regarding prevention of sexual harrassment and the company has provided information on the number of sexual harassment incidents."

    """
    
    ground_truth = """
    According to page 192 of annual_report_url.pdf, the company explicitly confirms the existence of a Prevention of Sexual Harassment (POSH) policy. The same page also reports that 11 complaints related to sexual harassment were filed during the financial year.
This indicates that the company has publicly disclosed both its POSH policy and the number of incidents, satisfying the criteria for the score 2 under the relevant evaluation rubric.
    """
    
    
    llm_output = """
    According to page 192 of annual_report_url.pdf, the company explicitly states the existence of a Prevention of Sexual Harassment (POSH) policy.  The same report, on page 192, also discloses that 11 complaints were filed during the financial year. Therefore, the company has both a POSH policy and has provided information on the number of sexual harassment incidents. Hence score 2 is given.
    """
    
    # Initialize evaluator
    evaluator = FactBasedEvaluator(
        model_name="gemini-1.5-flash",
        temperature=0.1
    )
    
    # Run evaluation
    print("🚀 Starting ultra-short fact-based evaluation...")
    summary = evaluator.evaluate(
        topic_definition=topic_definition,
        scoring_rubric=scoring_rubric,
        ground_truth=ground_truth,
        llm_output=llm_output
    )
    
    # Save results
    evaluator.save_results(summary)
    
    # Display final summary
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"   Total Questions: {summary.total_questions}")
    print(f"   Correct Facts: {summary.correct_facts}")
    print(f"   Accuracy: {summary.accuracy_percentage:.1f}%")
    print(f"   Result: {'✅ PASS' if summary.accuracy_percentage >= 70 else '❌ FAIL'}")

if __name__ == "__main__":
    main()
    
    
#############  Topic for AGM Delay Evaluation #############

    # # Example data
    # topic_definition = """
    # Question: Within how many months of the fiscal year end was the last AGM held?
    
    # Topic: Annual General Meeting Compliance Assessment
    # Goal: Evaluate whether the company held its AGM within the required timeframe after fiscal year end
    # Guidance: Analyze the timing between fiscal year end date and AGM date to determine compliance level
    # """
    
    # scoring_rubric = """
    # Score 0: More than six months after the fiscal year end
    # Score 1: Within four-six months of the fiscal year end  
    # Score 2: Within four months of the fiscal year end
    # """
    
    # ground_truth = """
    # The company scored 2 because the gap between FYE (March 31, 2024) and AGM (June 24, 2024) is less than 4 months. 
    # The calculation shows that there are 85 days between these two dates, which is less than 4 months. 
    # This information can be found on page 5 of the annual report, specifically in the section titled "About the Report" 
    # for the FYE, and on page 1 for the AGM notice.
    # Sources: pp. 1, 5 (annual_report.pdf)
    # """
    
    # llm_output = """
    # According to page 135 of annual_report_url.pdf, the company's financial year-end date is March 31, 2024.  The AGM is scheduled for June 24, 2024, as stated on page 478 of annual_report_url.pdf.  Research Finding states that there are 2 months and 23 days between March 31, 2024, and June 24, 2024 (page 9 of combined_policies.pdf). Another research finding confirms that the period between March 31, 2024, and June 24, 2024, is less than 4 months (various pages across multiple documents - this finding is supported by multiple references but does not provide specific page numbers in a way that allows me to cite them accurately in the requested format). Based on this information, and the provided scoring rubric, the gap between the financial year-end and the AGM is less than 4 months, resulting in a score of 2.
    # """
    
##################### Women Workforce Representation ####################

# topic_definition = """
#     Question: 
    
#     Topic: Women workforce representation assessment
#     Goal: To assess if company has sufficient women representation in workforce.
#     Guidance: You need to look for percentage of women in total workforce. This information is typically found in the corporate governance report or the annual report. If direct ratio is not given, try to look for total women employees and total number of employees and calculate ratio yourself. Make sure you quote this source in the answer with the page number from which you extract the information.
#     """
    
#     scoring_rubric = """
#         score 0: "if there is no such disclosure or the percentage of women in workforce is less than 10%",
#         score 1: "if percentage of  women in workforce is between 10% to 30%",
#         score 2: "if percentage of  women in workforce is more than 30%"
#     """
    
#     ground_truth = """
#     According to page 96 of annual_report_url.pdf, the percentage of women in Tata Motors' total workforce is 11.1%. The ratio is between 10'%' and 30'%' and hence score is 1 .
#     """
    
    
#     llm_output = """
#     According to page 96 of annual_report_url.pdf, the percentage of women in Tata Motors' total workforce is 11.1%.  Since this percentage falls between 10% and 30%, a score of 1 is assigned according to the provided rubric.
#     """
    
#################### Related Party Transactions Oversight ####################

#     topic_definition = """
#     Question: Does company have proper Related Party Transactions Oversight?
    
#     Topic: Related Party Transactions Oversight
#     Goal: To evaluate the governance framework for managing related party transactions
#     Guidance: You need to look for policy regarding related party transactions. We need to evaluate first if such a policy exists. If it does, is it comprehensive and well-defined? Make sure to quote this source in the answer with the page number from which you extract the information.
#     """
    
#     scoring_rubric = """
#         Score 0: "if policy regarding related party transactions is not present",
#         Score 1: "if policy regarding related party transactions is present but not comprehensive or not well-defined",
#         Score 2: "if policy regarding related party transactions is present and it is comprehensive and well-defined"
#     """
    
#     ground_truth = """
#     Tata Motors Limited has a comprehensive and well-defined policy on related party transactions, as documented in combined_policies.pdf and corroborated by other sources such as cg_report.pdf, annual_report_url.pdf, and credit_rating.pdf.

# Page 2 of combined_policies.pdf defines key terms including “related party,” “arm’s length transaction,” and “material modification.”

# Pages 3 and 5 detail procedures for identifying related parties and related transactions, the authorization process involving the Audit Committee and Board of Directors, thresholds requiring shareholder approval, and protocols for omnibus approvals.

# Page 3 outlines disclosure obligations under SEBI Listing Regulations and other applicable laws.

# Page 4 describes the Audit Committee’s responsibilities in reviewing related party transactions and ensuring periodic review and amendment of the policy.

# Page 3 also mentions the process of obtaining independent professional valuations.

# Page 5 specifies criteria for omnibus approvals.

# Page 8 mandates comprehensive disclosure practices.
# The documents consistently emphasize conflict-of-interest prevention, independent assessments, and transparency in approval and disclosure processes, thereby reinforcing the robustness of the company’s related party transaction policy.
#     """
    
#     llm_output = """
#     The research findings strongly indicate that Tata Motors Limited possesses a comprehensive and well-defined policy regarding related party transactions.  According to page 2 of combined_policies.pdf, the policy defines key terms such as \"related party,\" \"arm's length transaction,\" and \"material modification.\"  Pages 3 and 5 of combined_policies.pdf detail the processes for identifying related parties and transactions, and the authorization process involving the Audit Committee and Board of Directors, including thresholds for shareholder approval and procedures for omnibus approvals.  Page 3 of combined_policies.pdf outlines the policy's mandate for disclosures as per SEBI Listing Regulations and other laws.  Furthermore, page 4 of combined_policies.pdf highlights the Audit Committee's role in reviewing related party transactions and the policy's requirement for ongoing review and amendment.  The policy's emphasis on preventing conflicts of interest, ensuring independent valuations, and promoting transparency in authorization and disclosure is further supported by specific examples and details across multiple pages in the cited documents.  For example, page 3 of combined_policies.pdf describes the process for seeking external professional opinions for valuations. Page 5 of combined_policies.pdf outlines specific criteria for omnibus approvals. Page 8 of combined_policies.pdf mandates comprehensive disclosure. The consistent references to procedures and responsibilities across multiple documents (combined_policies.pdf, cg_report.pdf, annual_report_url.pdf, credit_rating.pdf)  further strengthens the evidence of a comprehensive and well-defined policy.
#     """

#####################   Board Independence   ####################

#   topic_definition = """
#      Question: Board Independence
    
#     Topic: Board Independence
#     Goal: To assess if the board have directors with permanent board seats
#     Guidance: You need to look for the corporate governance report. Find the reappointment date for each board members. If the reappointment date is either not provided or older than 5 years (i.e some date before 2019), then you need to check appointment date. If appointment date is also older than 5 years (i.e before 2019), mark that board member as permanent. Give list of board members and whether or not they are permanent. In other words, either of appointment date or reappointment date should be within last 5 years. For example, if a board member has appointment date '02-07-2020' and reappointment date is not present, then because the appointment date is within last 5 years (i.e March 2020 to March 2025 assuming we are checking for annual report as of 31st March 2025) then we would label them as 'Not permanent'. Second example, if any board member has appointment date as 01-01-2012 and reappointment date not present, then we would mark them permanent. Do not present output in table format. Give me text based paragraphs. You are looking at the corporate governance report as of 31st March 2024. Make sure you quote this source in the answer with the page number from which you extract the information.
#     """
    
#     scoring_rubric = """
#         Score 0: "if any one of the directors is marked as permanent board members as well as they are not explicitly mentioned to be representatives of lenders.",
#         Score 1: "if the directors which are marked as permanent board members, but those are representatives of lenders. Remember that usually this case is applicable for financially distressed companies. So unless it is mentioned explicitly that lenders have sent those board members as representative, do not assume so.",
#         Score 2: "if All directors are marked as non-permanent board members"
#     """
    
#     ground_truth = """
#     According to page 235 and page 17 of annual_report_url.pdf, multiple board members have appointment or most recent reappointment dates that precede March 31, 2019. These individuals are not identified as lender representatives.
# Specifically, page 147 of annual_report_url.pdf mentions Mr. Om Prakash Bhatt, whose most recent appointment was prior to March 31, 2019, and he is not a representative of lenders.
# This configuration—board members serving for more than five years without being lender representatives—triggers a score of 0 under the applicable scoring rubric.
#     """
    
    
#     llm_output = """
#     According to page 235 of annual_report_url.pdf, and corroborated by page 17 of the same document, there are several board members whose appointment or most recent reappointment date precedes March 31st, 2019.  These individuals are not explicitly identified as representatives of lenders.  Furthermore, page 147 of annual_report_url.pdf specifically mentions Mr. Om Prakash Bhatt, whose most recent appointment was before March 31st, 2019, and he is not a lender representative. The presence of these board members, whose appointments predate the 5-year threshold and who are not lender representatives, automatically results in a score of 0 according to the provided scoring rubric.
#     """
  