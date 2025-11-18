from src.api.routes.guardrails_template import GUARDRAILS_TEMPLATE
from src.api.routes.concern_risk_misc_naics import concerns_events, emerging_risks, misc_topics, naics_data
from src.api.routes.logger import get_logger
from src.api.routes.bedrock_client import BedrockClient, BedrockConfig
from src.api.routes.settings import BEDROCK_MODEL
import json
import re


logger = get_logger(__name__)


# RELEVANCE_CHECK_PROMPT = """
# You are an expert at determining if a query is relevant to the insurance industry and risk management domain.

# INDEX SCHEMA:
# {schema}

# DOMAIN CONTEXT:
# The system indexes articles related to:
# 1. Insurance Industry Topics: underwriting, claims, policies, premiums, coverage, reinsurance, actuarial analysis
# 2. Risk Management: emerging risks, risk assessment, risk mitigation, catastrophic events
# 3. Business Concerns: {concerns}
# 4. Emerging Risks: {emerging_risks}
# 5. Industry Topics: {misc_topics}
# 6. Industry Sectors (NAICS): {naics_descriptions}

# USER QUERY: {query}

# INSTRUCTIONS:
# 1. Analyze if the query is relevant to ANY of the following:
#    - Insurance industry operations, products, or services
#    - Risk management and emerging risks
#    - Business concerns listed above
#    - Industry sectors covered by NAICS codes
#    - Topics that would appear in insurance/risk management articles
   
# 2. Consider queries relevant if they ask about:
#    - Specific events/topics that affect insurance (e.g., "climate change", "wildfires", "PFAS")
#    - Industry sectors or business types
#    - Legal/regulatory matters affecting insurance
#    - Financial or operational risks
#    - General news that insurers would monitor
   
# 3. Consider queries NOT relevant if they are:
#    - Completely unrelated to business/insurance (e.g., "best pizza recipe")
#    - Personal queries unrelated to insurance (e.g., "my favorite color")
#    - Technical queries about the system itself (e.g., "how does OpenSearch work")
#    - Random or nonsensical queries

# 4. Calculate a relevance score between 0.0 and 1.0:
   
#    SCORE GUIDELINES:
#    - 0.9 - 1.0: Highly relevant - Direct insurance/risk management query
#      Examples: "Show me articles about property insurance claims", "PFAS contamination risks", "wildfire insurance coverage"
   
#    - 0.7 - 0.89: Moderately relevant - Related to business concerns or industries insurers monitor
#      Examples: "climate change impacts", "supply chain disruptions", "regulatory changes in healthcare"
   
#    - 0.5 - 0.69: Partially relevant - General business/economic topics that may affect insurance
#      Examples: "inflation trends", "labor market changes", "technology adoption in retail"
   
#    - 0.3 - 0.49: Marginally relevant - Tangentially related to business but unclear insurance connection
#      Examples: "social media trends", "consumer behavior patterns", "general market news"
   
#    - 0.0 - 0.29: Not relevant - Unrelated to insurance, risk management, or business
#      Examples: "best pizza recipe", "how to play guitar", "movie recommendations"
   
#    SET is_relevant to true if score >= 0.5, false if score < 0.5

# 5. Respond with ONLY a JSON object in this exact format:
# {{
#   "is_relevant": True or False,
#   "reason": "Brief explanation of why the query is or isn't relevant and how you determined the score",
#   "score": 0.85
# }}

# CRITICAL: Return ONLY the JSON object, no other text or markdown.
# """


OPENSEARCH_SCHEMA = """ 
Index Name: ei_articles_index

Field Mappings (exact names from OpenSearch) along with their type of search to be performed for retrieval:
- title (text)
- data (text)
- description (text)
- reason_identified (text)
- published_time (text)
- last_update_time (text)
- injection_time (text)
- is_latest (boolean)
- url (keyword)
- concerns (text)
- emerging_risk_name (text)
- region (keyword)
- miscTopics (text)
- naicscode (keyword)
- naics_description (text)
- source (keyword)
- tag (keyword)
- doc_id (long)
- source_meta (object: {{rss_entry (text), title (text)}})
- chunk_id (integer)
- field (text) — field name from which chunk is derived
- chunk_text (text) — text chunk content
- chunk_vector (knn_vector, 1024-dim) — embedding vector for chunk_text

Vector Fields (for semantic search):
- chunk_vector (used for semantic similarity search across title, data, and reason_identified)

Available Values:
- Tags: Current, Potential New Trend, Untagged, Processing Error
"""


# OPENSEARCH_SCHEMA = """ 
# Index Name: ei_articles_index

# Field Mappings (exact names from OpenSearch):
# - title (text)
# - data (text)
# - description (text)
# - reason_identified (text)
# - published_time (text)
# - last_update_time (text)
# - injection_time (text)
# - is_latest (boolean)
# - url (keyword)
# - concerns (keyword)
# - emerging_risk_name (keyword)
# - region (keyword)
# - miscTopics (keyword)
# - naicscode (keyword)
# - naics_description (keyword)
# - source (keyword)
# - tag (keyword)
# - doc_id (long)
# - source_meta (object: {{rss_entry (text), title (text)}})
# - chunk_id (integer)
# - field (text) — field name from which chunk is derived
# - chunk_text (text) — text chunk content
# - chunk_vector (knn_vector, 1024-dim) — embedding vector for chunk_text

# Vector Fields (for semantic search):
# - chunk_vector (used for semantic similarity search across title, data, and reason_identified)

# Available Values:
# - Tags: Current, Potential New Trend, Untagged, Processing Error
# """

def _prepare_schema() -> str:
    """Prepare schema for the prompt."""
    return OPENSEARCH_SCHEMA.format(
        concerns=", ".join(concerns_events) + "...",
        emerging_risks=", ".join(emerging_risks) + "...",
        misc_topics=", ".join(misc_topics) + "...",
        naics_data=", ".join([f"{item['code']}" for item in naics_data]) + "..."
    )

# RELEVANCE_CHECK_PROMPT = """
# You are an expert at determining if a query is relevant to the insurance industry and risk management domain.

# DOMAIN CONTEXT:
# The system indexes articles related to:
# 1. Insurance Industry Topics: underwriting, claims, policies, premiums, coverage, reinsurance, actuarial analysis
# 2. Risk Management: emerging risks, risk assessment, risk mitigation, catastrophic events
# 3. Business Concerns: {concerns}
# 4. Emerging Risks: {emerging_risks}
# 5. Industry Topics: {misc_topics}
# 6. Industry Sectors (NAICS): {naics_descriptions}

# USER QUERY: {query}

# INSTRUCTIONS:
# 1. Analyze if the query is relevant to ANY of the following:
#    - Insurance industry operations, products, or services
#    - Risk management and emerging risks
#    - Business concerns listed above
#    - Industry sectors covered by NAICS codes
#    - Topics that would appear in insurance/risk management articles
   
# 2. Consider queries relevant if they ask about:
#    - Specific events/topics that affect insurance (e.g., "climate change", "wildfires", "PFAS")
#    - Industry sectors or business types
#    - Legal/regulatory matters affecting insurance
#    - Financial or operational risks
#    - General news that insurers would monitor
   
# 3. Consider queries NOT relevant if they are:
#    - Completely unrelated to business/insurance (e.g., "best pizza recipe")
#    - Personal queries unrelated to insurance (e.g., "my favorite color")
#    - Technical queries about the system itself (e.g., "how does OpenSearch work")
#    - Random or nonsensical queries

# 4. Score shoul be calculated 

# 5. Respond with ONLY a JSON object in this exact format:
# {{
#   "is_relevant": true or false,
#   "reason": "Brief explanation of why the query is or isn't relevant",
#   "score": "how relevant the query [score must be in between 0 and 1]"
# }}

# CRITICAL: Return ONLY the JSON object, no other text.
# """

# Add this new method to the OpenSearchQueryGenerator class

def _extract_json(text: str) -> dict:
        """Extract JSON from the LLM's response text with robust handling."""
        text = text.strip()
        logger.debug(f"Raw LLM output (first 500 chars): {text[:500]}")

        # Common cleanup for markdown/code fences
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
        text = text.strip("` \n\t")

        # Try to find JSON object anywhere in text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            logger.error("No JSON braces found in response.")
            return {}

        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {e}. Candidate snippet: {candidate[:300]}")
            return {}

async def check_query_relevance( user_query: str) -> tuple[bool, str]:
    """
    Check if the user query is relevant to insurance/risk management domain.
    
    Args:
        user_query: The natural language query to check
        
    Returns:
        tuple: (is_relevant: bool, reason: str)
    """
    bedrock = BedrockClient(BedrockConfig())
    try:
        schema = _prepare_schema()
        # Prepare domain context
        # concerns_str = ", ".join(concerns_events)  # First 20 items
        # risks_str = ", ".join(emerging_risks)
        # misc_str = ", ".join(misc_topics)
        # naics_str = ", ".join([f"{item['description']}" for item in naics_data])
        
        # prompt = RELEVANCE_CHECK_PROMPT.format(
        #     schema=schema,
        #     query=user_query,
        #     concerns=concerns_str,
        #     emerging_risks=risks_str,
        #     misc_topics=misc_str,
        #     naics_descriptions=naics_str
        # )
        prompt = GUARDRAILS_TEMPLATE.format(user_input = user_query)
        
        logger.info(f"Checking relevance for query: {user_query}")
        model_id = BEDROCK_MODEL
        
        response_text = bedrock.invoke_model(
                model_id=model_id,
                prompt=prompt,
                max_tokens=100,
                temperature=0.3
            )
        print("response-----------------",response_text, type(response_text))
        logger.info(f"Relevance check response: {response_text}")

        if response_text.strip().lower() == "no":
            return True
        else:
            return False
        
        # Extract JSON from response
        # result = _extract_json(response_text)
        
        # if not result or "is_relevant" not in result:
        #     logger.warning(f"Invalid relevance check response: {response_text}")
        #     # Default to True to avoid blocking valid queries on parsing errors
        #     return True, "Unable to determine relevance, proceeding with query", 0
        
        # is_relevant = result.get("is_relevant", True)
        # reason = result.get("reason", "No reason provided")
        # score = result.get("score", 0)
        
        # logger.info(f"Query relevance: {is_relevant}, Reason: {reason}")
        
        # return is_relevant, reason, score
        
    except Exception as e:
        logger.error(f"Error checking query relevance: {e}", exc_info=True)
        # Default to True on error to avoid blocking valid queries
        return True, f"Error during relevance check: {str(e)}", 0
