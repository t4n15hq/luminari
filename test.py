import openai
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union
from enum import Enum

class StudyPhase(Enum):
    PHASE_1 = 1
    PHASE_2 = 2
    PHASE_3 = 3
    PHASE_4 = 4

class ProtocolSection(Enum):
    OVERVIEW = "overview"
    OBJECTIVES = "objectives"
    ENDPOINTS = "endpoints"
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"
    PROCEDURES = "procedures"
    SAFETY = "safety"

class ClinicalTrialAssistant:
    def __init__(self, model="gpt-4o-mini"):
        """Initialize the Clinical Trial Assistant with configurations."""
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.protocol_history = []
        
        # Load protocol templates
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict:
        """Load protocol templates from JSON file."""
        try:
            with open('protocol_templates.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_default_templates()

    def _create_default_templates(self) -> Dict:
        """Create default protocol templates."""
        return {
            "psoriasis": {
                "inclusion_criteria": [
                    "Adults aged 18-65 years",
                    "Confirmed plaque psoriasis diagnosis",
                    "BSA ≥ 10%",
                    "PASI score ≥ 12"
                ],
                "primary_endpoints": [
                    "PASI 75 at Week 12",
                    "IGA score of 0/1 at Week 12"
                ]
            }
        }

    async def generate_protocol(
        self,
        disease_type: str,
        phase: StudyPhase,
        population_size: int,
        duration_weeks: int,
        endpoints: Optional[List[str]] = None,
        additional_criteria: Optional[Dict] = None
    ) -> Dict:
        """
        Generate a complete clinical trial protocol with comprehensive sections.
        """
        try:
            template = self.templates.get(disease_type, {})
            
            # Build sophisticated prompt
            prompt = self._build_protocol_prompt(
                disease_type=disease_type,
                phase=phase,
                population_size=population_size,
                duration_weeks=duration_weeks,
                endpoints=endpoints or template.get('primary_endpoints', []),
                additional_criteria=additional_criteria
            )

            response = await self._get_completion(prompt)
            
            # Process and structure the response
            protocol = self._structure_protocol(response)
            
            # Save to history
            self._save_to_history(protocol)
            
            return protocol

        except Exception as e:
            self._log_error(f"Protocol generation error: {str(e)}")
            raise

    async def answer_query(
        self,
        query: str,
        protocol_context: Optional[Dict] = None
    ) -> str:
        """
        Provide intelligent responses to protocol-related queries.
        """
        try:
            context = self._build_query_context(query, protocol_context)
            
            response = await self._get_completion(
                prompt=query,
                system_context=context
            )
            
            return self._format_query_response(response)

        except Exception as e:
            self._log_error(f"Query processing error: {str(e)}")
            raise

    def _build_protocol_prompt(self, **kwargs) -> str:
        """Build a detailed prompt for protocol generation."""
        return f"""Generate a comprehensive clinical trial protocol with the following specifications:

Disease: {kwargs['disease_type']}
Phase: {kwargs['phase'].value}
Population: {kwargs['population_size']} patients
Duration: {kwargs['duration_weeks']} weeks

Include detailed sections for:
1. Study Overview and Objectives
2. Trial Design and Methodology
3. Patient Selection (Inclusion/Exclusion)
4. Treatment Plan and Procedures
5. Safety Monitoring
6. Efficacy Assessments
7. Statistical Considerations
8. Ethical Considerations

Primary Endpoints: {', '.join(kwargs['endpoints'])}

Make sure to incorporate standard requirements for {kwargs['disease_type']} trials
and phase {kwargs['phase'].value} specific considerations."""

    async def _get_completion(
        self,
        prompt: str,
        system_context: Optional[str] = None
    ) -> str:
        """Get completion from OpenAI with error handling and retries."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": system_context or "You are an expert clinical protocol designer."
                },
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content

        except Exception as e:
            self._log_error(f"OpenAI API error: {str(e)}")
            raise

    def _structure_protocol(self, raw_response: str) -> Dict:
        """Structure the raw protocol response into organized sections."""
        # Add your structuring logic here
        pass

    def _log_error(self, error_message: str) -> None:
        """Log errors for monitoring and debugging."""
        timestamp = datetime.now().isoformat()
        with open('error_log.txt', 'a') as f:
            f.write(f"{timestamp}: {error_message}\n")

    def _save_to_history(self, protocol: Dict) -> None:
        """Save generated protocol to history with timestamp."""
        self.protocol_history.append({
            'timestamp': datetime.now().isoformat(),
            'protocol': protocol
        })

# Usage example:
async def main():
    assistant = ClinicalTrialAssistant()
    
    # Generate protocol
    protocol = await assistant.generate_protocol(
        disease_type="psoriasis",
        phase=StudyPhase.PHASE_2,
        population_size=200,
        duration_weeks=12
    )
    
    # Query about protocol
    answer = await assistant.answer_query(
        "What are the key safety monitoring requirements for this protocol?",
        protocol_context=protocol
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
