import json
import re
from typing import Optional, List
from src.api.routes.bedrock_client import BedrockClient, BedrockConfig
from src.api.routes.logger import get_logger
from src.api.routes.concern_risk_misc_naics import concerns_events, emerging_risks, misc_topics, naics_data
from src.api.routes.settings import BEDROCK_MODEL
from CommonService.CommonService.async_bedrock.base import TitanV2
import asyncio
from CommonService.async_commonsession.commonsession import (
            CommonSession,
            CommonSessionConfig,
        )
from src.api.routes.validate_user_query import check_query_relevance

logger = get_logger(__name__)


# ------------------ OPENSEARCH SCHEMA ------------------

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

# ------------------ PROMPT TEMPLATE ------------------


QUERY_GENERATION_PROMPT = """
You are an expert at converting natural language queries into valid OpenSearch DSL queries with HYBRID search (both semantic and keyword).

INDEX SCHEMA:
{schema}

USER QUERY: {query}

CRITICAL INSTRUCTIONS:
1. Generate ONLY a valid JSON object for the OpenSearch query body.
2. Use EXACT field names from the schema (case-sensitive).
3. HYBRID SEARCH DEFAULT RULE:
   - By default, for normal queries (without quotes), always use HYBRID search:
       → knn (chunk_vector) + multi_match on text fields.
4. OVERRIDE RULE — HIGHEST PRIORITY (CRITICAL):
   ***This rule overrides ALL other rules, including hybrid search.***
   - If the user highlights (specifies) ANY text inside single quotes ('...')
     or double quotes ("..."), then:
       → DO NOT generate knn
       → DO NOT perform semantic search
       → ONLY use keyword-based search (match, multi_match, term, range)
   - This rule ALWAYS wins even if the query contains topics, events,
     or anything normally handled with hybrid search.
5. never consider region field even it explicity specified in user query
6. Field types:
   - Keyword: tag, source,  naicscode, url
   - Text (for keyword search): title, description, data, reason_identified, chunk_text, concerns, emerging_risk_name, miscTopics, naics_description
   - Text (for keyword search — use `match` or `multi_match`, never `term`): 
      title, description, data, reason_identified, chunk_text, concerns, emerging_risk_name, miscTopics, naics_description
6. Field types:
   - Keyword: tag, source,  naicscode, url
   - Text (for keyword search): title, description, data, reason_identified, chunk_text, concerns, emerging_risk_name, miscTopics, naics_description
   - Boolean: is_latest
   - Numeric: doc_id, chunk_id
   - Date/time: published_time, last_update_time, injection_time

7. Hybrid search structure (ALWAYS use this for content queries):
   {{
     "query": {{
       "bool": {{
         "must": [
           {{
             "knn": {{
               "chunk_vector": {{
                 "vector": "__EMBEDDING_TEXT__",
                 "k": 10
               }}
             }}
           }},
           {{
             "multi_match": {{
               "query": "__KEYWORD_TEXT__",
               "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified", "concerns", "emerging_risk_name", "miscTopics"],
               "type": "best_fields",
               "operator": "or"
             }}
           }}
         ]
       }}
     }}
   }}

8. Boosting strategy:
   - title^3 (highest relevance)
   - data^2, chunk_text^2 (medium relevance)
   - reason_identified (base relevance)

9. Combination examples:
   - "Articles about climate change":
     → Hybrid search (KNN + multi_match) on "climate change"
   
   - "Articles about climate change tagged as Current":
     → bool.must with: KNN, multi_match, and term(tag="Current")
   
   - "Show PFAS articles":
     → Hybrid search on "PFAS" + term(concerns="PFAS")
   
   - "Articles about wildfires in last 3 days":
     → bool.must with: KNN, multi_match, and range on published_time

10. Return ONLY the JSON object — no text, markdown, or explanations.
11. values for reference:
    - concerns : {concerns}
    - emerging_risk_name : {emerging_risk_name}
    - misc_topics : {misc_topics}
    - naicscode and naics_description : {naics}
---

EXAMPLES:

Query: "Show me all articles tagged as Current"
{{
  "query": {{
    "term": {{
      "tag": "Current"
    }}
  }}
}}

Query: "Show all articles on 'new reseach' "
{{
  "query": {{
    "bool": {{
      "must": [
        {{
          "multi_match": {{
            "query": "new research",
            "fields": [
              "title^3",
              "data^2",
              "chunk_text^2",
              "reason_identified",
              "concerns",
              "emerging_risk_name",
              "miscTopics",
              "naics_description"
            ],
            "type": "phrase"
          }}
        }}
      ]
    }}
  }}
}}


Query: "Find articles about climate change"
{{
  "query": {{
    "bool": {{
      "must": [
        {{
          "knn": {{
            "chunk_vector": {{
              "vector": "__EMBEDDING_TEXT__",
              "k": 10
            }}
          }}
        }},
        {{
          "multi_match": {{
            "query": "__KEYWORD_TEXT__",
            "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified", "concerns", "emerging_risk_name", "miscTopics"],
            "type": "best_fields",
            "operator": "or"
          }}
        }}
      ]
    }}
  }}
}}

Query: "Find articles about lawsuits tagged as Current"
{{
  "query": {{
    "bool": {{
      "must": [
        {{
          "term": {{
            "tag": "Current"
          }}
        }},
        {{
          "knn": {{
            "chunk_vector": {{
              "vector": "__EMBEDDING_TEXT__",
              "k": 10
            }}
          }}
        }},
        {{
          "multi_match": {{
            "query": "__KEYWORD_TEXT__",
            "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified", "concerns", "emerging_risk_name", "miscTopics"],
            "type": "best_fields",
            "operator": "or"
          }}
        }}
      ]
    }}
  }}
}}

Query: "Show untagged articles"
{{
  "query": {{
    "term": {{
      "tag": "Untagged"
    }}
  }}
}}

Query: "Show articles about wildfire in last 3 days"
{{
  "query": {{
    "bool": {{
      "must": [
        {{
          "range": {{
            "published_time": {{
              "gte": "now-3d/d",
              "lte": "now"
            }}
          }}
        }},
        {{
          "knn": {{
            "chunk_vector": {{
              "vector": "__EMBEDDING_TEXT__",
              "k": 10
            }}
          }}
        }},
        {{
          "multi_match": {{
            "query": "__KEYWORD_TEXT__",
            "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified", "concerns", "emerging_risk_name", "miscTopics"],
            "type": "best_fields",
            "operator": "or"
          }}
        }}
      ]
    }}
  }}
}}
"""




# QUERY_GENERATION_PROMPT = """
# You are an expert at converting natural language queries into valid OpenSearch DSL queries with HYBRID search (both semantic and keyword).

# INDEX SCHEMA:
# {schema}

# USER QUERY: {query}

# CRITICAL INSTRUCTIONS:
# 1. Generate ONLY a valid JSON object for the OpenSearch query body.
# 2. Use EXACT field names from the schema (case-sensitive).
# 3. ALWAYS use HYBRID search combining both semantic (KNN) and keyword search for content queries.
# 4. Query rules:
#    - QUOTED TEXT RULE (CRITICAL) ---
#       - If the user highlights (specifies) any text inside single quotes ('...')
#          or double quotes ("..."), then vector search MUST NOT be used.
#       - In such cases:
#        → DO NOT generate a knn clause
#        → ONLY use match / multi_match / term / range queries depending on field type.
#       - Quoted text means the user wants exact or phrase-based keyword search.
#    - For content/topic queries (about articles, topics, events):
#        → Use HYBRID search with BOTH:
#          a) KNN semantic search on `chunk_vector`
#          b) Multi-match keyword search on text fields (title, data, reason_identified)
#        → Use the placeholder text "__EMBEDDING_TEXT__" for semantic search
#        → Use the placeholder text "__KEYWORD_TEXT__" for keyword search
#    - For text fields like `concerns`, `emerging_risk_name`, `miscTopics`, and `naics_description`:
#        → ALWAYS use `match` queries (or include them inside multi_match)
#        → NEVER use `term` queries, even if the query term matches exactly.
#        → Example: if user query mentions "injury" and the field has "injuries", `match` must retrieve it (since analyzer performs stemming).
       
#    - For metadata-only searches (e.g. tag, source ):
#        → Use only `term`, `terms`, `match`, or `range` queries on those fields.
   
#    - To combine hybrid search with filters:
#        → Use a `bool` query with `must` containing both KNN, multi_match, and filters.
# 5. never consider region field even it explicity specified in user query
# 6. Field types:
#    - Keyword: tag, source,  naicscode, url
#    - Text (for keyword search): title, description, data, reason_identified, chunk_text, concerns, emerging_risk_name, miscTopics, naics_description
#    - Text (for keyword search — use `match` or `multi_match`, never `term`): 
#       title, description, data, reason_identified, chunk_text, concerns, emerging_risk_name, miscTopics, naics_description
# 6. Field types:
#    - Keyword: tag, source,  naicscode, url
#    - Text (for keyword search): title, description, data, reason_identified, chunk_text, concerns, emerging_risk_name, miscTopics, naics_description
#    - Boolean: is_latest
#    - Numeric: doc_id, chunk_id
#    - Date/time: published_time, last_update_time, injection_time

# 7. Hybrid search structure (ALWAYS use this for content queries):
#    {{
#      "query": {{
#        "bool": {{
#          "must": [
#            {{
#              "knn": {{
#                "chunk_vector": {{
#                  "vector": "__EMBEDDING_TEXT__",
#                  "k": 10
#                }}
#              }}
#            }},
#            {{
#              "multi_match": {{
#                "query": "__KEYWORD_TEXT__",
#                "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified", "concerns", "emerging_risk_name", "miscTopics"],
#                "type": "best_fields",
#                "operator": "or"
#              }}
#            }}
#          ]
#        }}
#      }}
#    }}

# 8. Boosting strategy:
#    - title^3 (highest relevance)
#    - data^2, chunk_text^2 (medium relevance)
#    - reason_identified (base relevance)

# 9. Combination examples:
#    - "Articles about climate change":
#      → Hybrid search (KNN + multi_match) on "climate change"
   
#    - "Articles about climate change tagged as Current":
#      → bool.must with: KNN, multi_match, and term(tag="Current")
   
#    - "Show PFAS articles":
#      → Hybrid search on "PFAS" + term(concerns="PFAS")
   
#    - "Articles about wildfires in last 3 days":
#      → bool.must with: KNN, multi_match, and range on published_time

# 10. Return ONLY the JSON object — no text, markdown, or explanations.
# 11. values for reference:
#     - concerns : {concerns}
#     - emerging_risk_name : {emerging_risk_name}
#     - misc_topics : {misc_topics}
#     - naicscode and naics_description : {naics}
# ---

# EXAMPLES:

# Query: "Show me all articles tagged as Current"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Current"
#     }}
#   }}
# }}

# Query: "Show all articles on 'new reseach' "
# {{
#   "query": {{
#     "term": {{
#       "concern": "new research"
#     }}
#   }}
# }}

# Query: "Find articles about climate change"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "knn": {{
#             "chunk_vector": {{
#               "vector": "__EMBEDDING_TEXT__",
#               "k": 10
#             }}
#           }}
#         }},
#         {{
#           "multi_match": {{
#             "query": "__KEYWORD_TEXT__",
#             "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified", "concerns", "emerging_risk_name", "miscTopics"],
#             "type": "best_fields",
#             "operator": "or"
#           }}
#         }}
#       ]
#     }}
#   }}
# }}

# Query: "Find articles about lawsuits tagged as Current"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "term": {{
#             "tag": "Current"
#           }}
#         }},
#         {{
#           "knn": {{
#             "chunk_vector": {{
#               "vector": "__EMBEDDING_TEXT__",
#               "k": 10
#             }}
#           }}
#         }},
#         {{
#           "multi_match": {{
#             "query": "__KEYWORD_TEXT__",
#             "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified", "concerns", "emerging_risk_name", "miscTopics"],
#             "type": "best_fields",
#             "operator": "or"
#           }}
#         }}
#       ]
#     }}
#   }}
# }}

# Query: "Show untagged articles"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Untagged"
#     }}
#   }}
# }}

# Query: "Show articles about wildfire in last 3 days"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "range": {{
#             "published_time": {{
#               "gte": "now-3d/d",
#               "lte": "now"
#             }}
#           }}
#         }},
#         {{
#           "knn": {{
#             "chunk_vector": {{
#               "vector": "__EMBEDDING_TEXT__",
#               "k": 10
#             }}
#           }}
#         }},
#         {{
#           "multi_match": {{
#             "query": "__KEYWORD_TEXT__",
#             "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified", "concerns", "emerging_risk_name", "miscTopics"],
#             "type": "best_fields",
#             "operator": "or"
#           }}
#         }}
#       ]
#     }}
#   }}
# }}
# """


class OpenSearchQueryGenerator:
    def __init__(self):
        self.bedrock = BedrockClient(BedrockConfig())
        self.model_id = BEDROCK_MODEL
        self.embedding_model_id = "amazon.titan-embed-text-v2:0"

    def _prepare_schema(self) -> str:
        """Prepare schema for the prompt."""
        return OPENSEARCH_SCHEMA.format(
            concerns=", ".join(concerns_events) + "...",
            emerging_risks=", ".join(emerging_risks) + "...",
            misc_topics=", ".join(misc_topics) + "...",
            naics_data=", ".join([f"{item['code']}" for item in naics_data]) + "..."
        )
    
    async def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embedding vector for given text using Amazon Titan Embedding model.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            List of floats representing the embedding vector (1024 dimensions for Titan v2)
        """
        try:
            logger.info(f"Generating embedding for text: {text[:100]}...")
            
            async with CommonSession(
                CommonSessionConfig(
                    client_name="bedrock-runtime",
                    region="us-east-1",
                    profile_name="Comm-Prop-Sandbox"
                )
            ) as titan:
                titan_embedding = TitanV2(titan)
                
                # ✅ CRITICAL: Pass dict, not JSON string
                # invoke_with_retry will call json.dumps() internally (base.py line 70)
                payload = {"inputText": text}
                response = await titan_embedding.generate_embedding(payload)
                
                # TitanV2.generate_embedding returns the embedding vector directly
                # as a list of floats (see base.py line 156 and line 72)
                if not isinstance(response, list):
                    logger.error(f"Unexpected response type: {type(response)}, value: {response}")
                    raise ValueError(f"Expected list of floats, got {type(response)}")
                
                if len(response) == 0:
                    raise ValueError("Received empty embedding vector")
                
                logger.info(f"✅ Successfully generated {len(response)}-dimensional embedding")
                return response
                
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            raise
    
    async def _create_hybrid_query(self, user_query: str, min_score: Optional[float] = None):
        """
        Build a hybrid query combining both semantic (KNN) and keyword (multi_match) search.
        
        Args:
            user_query: The search query text
            min_score: Optional minimum score threshold for results
        """
        embedding_vector = await self.generate_embeddings(user_query)
        logger.info(f"Generated embedding vector type: {type(embedding_vector)}, length: {len(embedding_vector)}")
        
        # ✅ Hybrid search: KNN + multi_match with relevance scoring
        query = {
            "size": 500,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "chunk_vector": {
                                    "vector": embedding_vector,
                                    "k": 10,
                                    "boost": 2.0  # Boost semantic similarity
                                }
                            }
                        },
                        {
                            "multi_match": {
                                "query": user_query,
                                "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified", "concerns", "emerging_risk_name", "miscTopics"],
                                "type": "best_fields",
                                "operator": "or",
                                "boost": 1.0  # Standard keyword search weight
                            }
                        }
                    ]
                }
            }
        }
        
        # Add minimum score filter if provided
        if min_score is not None:
            query["min_score"] = min_score
            logger.info(f"Applied min_score filter: {min_score}")
        
        return query

    def _needs_semantic_search(self, user_query: str) -> bool:
        """
        Determine whether the user query should trigger a semantic (vector) search.
        Returns True if the query likely targets 'data', 'reason_identified', or 'title'
        fields, or contains general article-topic language.
        """
        if not user_query:
            return False

        query_lower = user_query.lower()
        
        # Metadata-only keywords that don't need semantic search
        metadata_only_keywords = [
            "show all", "tagged as", "from source", "in region",
            "with tag", "untagged", "current", "potential new trend"
        ]
        
        # Check if query is purely metadata-based
        for keyword in metadata_only_keywords:
            if keyword in query_lower and "about" not in query_lower and "related" not in query_lower:
                return False
        
        # Semantic search keywords
        semantic_keywords = [
            "about", "related to", "discuss", "concerning",
            "impact", "risk", "article", "topic", "find",
            "search for", "looking for", "show me articles on"
        ]

        # If any semantic keyword appears, we need semantic search
        return any(keyword in query_lower for keyword in semantic_keywords)

    async def _add_hybrid_query(self, user_query: str, k: int = 10):
        """
        Add hybrid search (KNN + multi_match) if semantic search is required but missing.
        """
        logger.info("No hybrid search found but content search needed, adding hybrid query")

        embedding_vector = await self.generate_embeddings(user_query)
        if not isinstance(embedding_vector, list):
            raise ValueError("Embedding vector must be list of floats")

        hybrid_query = {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "chunk_vector": {
                                "vector": embedding_vector,
                                "k": k
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": user_query,
                            "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified", "concerns", "emerging_risk_name", "miscTopics"],
                            "type": "best_fields",
                            "operator": "or"
                        }
                    }
                ]
            }
        }

        logger.info(f"✅ Added fallback hybrid query for: {user_query}")
        return {"query": hybrid_query, "size": 500}

    async def generate_query(self, user_query: str, source_filter: Optional[str] = None, 
                           relevance_threshold: float = 0.5) -> dict:
        """
        Generate an OpenSearch query DSL body from a natural language query.
        
        Args:
            user_query: Natural language search query
            source_filter: Optional source filter (e.g., 'reuters', 'other')
            relevance_threshold: Minimum relevance threshold (0.0-1.0). 
                               0.5 means only results with score >= 50% of max score
        """
        try:
            is_relevant = await check_query_relevance(user_query)
            
            if not is_relevant:
                print(is_relevant)
                logger.warning(f"Query rejected as not relevant: {user_query}")
                # logger.warning(f"Rejection reason: {reason}")
                # logger.warning(f"score: {score}")
                return None
                # return {
                #     "error": "Query not relevant to insurance domain",
                #     "reason": reason,
                #     "query": user_query
                # }
            
            # logger.info(f"Query passed relevance check: {reason} with the score {score}")
            logger.info(f"Query passed relevance chec")
            schema = self._prepare_schema()
            prompt = QUERY_GENERATION_PROMPT.format(schema=schema, query=user_query, concerns = concerns_events,  emerging_risk_name = emerging_risks, misc_topics = misc_topics, naics = naics_data )

            # Get LLM to generate query structure
            response_text = self.bedrock.invoke_model(
                model_id=self.model_id,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.0
            )

            logger.info(f"=== LLM FULL Response ===\n{response_text}\n=== END Response ===")
            
            query_body = self._extract_json(response_text)
            logger.info(f"=== Extracted query_body ===\n{json.dumps(query_body, indent=2)}\n=== END query_body ===")

            # Check if we need semantic search
            needs_semantic_search = self._needs_semantic_search(user_query)
            logger.info(f"User query needs semantic/content search: {needs_semantic_search}")

            if not query_body or ("query" not in query_body and "knn" not in query_body):
                logger.warning(f"Invalid query generated. Response: {response_text[:200]}")
                if needs_semantic_search:
                    logger.info("Generating hybrid query directly as fallback")
                    query_body = await self._create_hybrid_query(user_query)
                else:
                    query_body = self._default_query()

            # Process placeholders - find and replace __EMBEDDING_TEXT__ and __KEYWORD_TEXT__
            try:
                processed = await self._process_placeholders(query_body, user_query)
                logger.info(f"Placeholders processed: {processed}")
                
                # If no hybrid search was found but query needs it, add it
                if not processed and needs_semantic_search:
                    logger.info("No hybrid search found but content search needed, adding hybrid query")
                    query_body = await self._create_hybrid_query(user_query)
                    
            except Exception as e:
                logger.error(f"Failed to process placeholders: {e}")
                if needs_semantic_search:
                    logger.info("Falling back to direct hybrid query creation")
                    query_body = await self._create_hybrid_query(user_query)
                else:
                    return self._default_query()

            # Ensure size parameter is set
            if "size" not in query_body:
                query_body["size"] = 500
            print("inside generate query, dhruv...........................", source_filter)
            # Add source filter if provided
            if source_filter:
                print("into source filter................")
                query_body = self._apply_source_filter(query_body, source_filter)

            logger.info(f"Successfully generated OpenSearch query for: {user_query}")
            if source_filter:
                logger.info(f"Source filter applied: {source_filter}")
            logger.info(f"=== FINAL Query body ===\n{json.dumps(query_body, indent=2)}\n=== END FINAL ===")
            return query_body

        except Exception as e:
            logger.error(f"Error generating OpenSearch query: {e}", exc_info=True)
            return self._default_query()

    def _extract_json(self, text: str) -> dict:
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

    async def _process_placeholders(self, query_body: dict, user_query: str) -> bool:
        """
        Recursively find and replace both __EMBEDDING_TEXT__ and __KEYWORD_TEXT__ placeholders.
        
        Returns:
            bool: True if at least one placeholder was found and processed
        """
        placeholder_found = False
        
        async def process_dict(obj, parent_key=None):
            nonlocal placeholder_found
            
            if isinstance(obj, dict):
                # Process KNN queries with __EMBEDDING_TEXT__
                if "knn" in obj:
                    knn_query = obj["knn"]
                    if isinstance(knn_query, dict):
                        await self._replace_knn_placeholder(knn_query, user_query)
                        placeholder_found = True
                
                # Process multi_match queries with __KEYWORD_TEXT__
                if "multi_match" in obj:
                    multi_match_query = obj["multi_match"]
                    if isinstance(multi_match_query, dict):
                        self._replace_keyword_placeholder(multi_match_query, user_query)
                        placeholder_found = True
                
                # Recursively process all nested structures
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        await process_dict(value, key)
            
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        await process_dict(item, parent_key)
        
        await process_dict(query_body)
        return placeholder_found

    async def _replace_knn_placeholder(self, knn_query: dict, user_query: str) -> None:
        """
        Replace __EMBEDDING_TEXT__ placeholder in KNN query with actual embedding vector.
        """
        try:
            # KNN query structure: {"chunk_vector": {"vector": "...", "k": 10}}
            for field_name, field_config in knn_query.items():
                if isinstance(field_config, dict) and "vector" in field_config:
                    vector_value = field_config["vector"]
                    
                    # Check if it's a placeholder or text that needs embedding
                    if isinstance(vector_value, str):
                        if "__EMBEDDING_TEXT__" in vector_value:
                            embedding_text = user_query
                            logger.info(f"Found __EMBEDDING_TEXT__ placeholder, using user query: {user_query}")
                        else:
                            embedding_text = vector_value
                            logger.info(f"Found text in vector field: {embedding_text}")
                        
                        # Generate the actual embedding vector
                        embedding_vector = await self.generate_embeddings(embedding_text)
                        
                        # Replace with actual vector
                        field_config["vector"] = embedding_vector
                        
                        logger.info(f"✅ Successfully replaced KNN placeholder with {len(embedding_vector)}-dim embedding vector")
                    elif isinstance(vector_value, list):
                        # Already has a vector, skip
                        logger.info(f"Field {field_name} already has embedding vector")
                        continue
            
        except Exception as e:
            logger.error(f"Failed to replace KNN placeholder: {e}", exc_info=True)
            raise

    def _replace_keyword_placeholder(self, multi_match_query: dict, user_query: str) -> None:
        """
        Replace __KEYWORD_TEXT__ placeholder in multi_match query with actual search text.
        """
        try:
            if "query" in multi_match_query:
                query_value = multi_match_query["query"]
                
                if isinstance(query_value, str) and "__KEYWORD_TEXT__" in query_value:
                    multi_match_query["query"] = user_query
                    logger.info(f"✅ Successfully replaced __KEYWORD_TEXT__ with: {user_query}")
                elif isinstance(query_value, str):
                    # Already has text, might be fine
                    logger.info(f"multi_match already has query text: {query_value}")
            
        except Exception as e:
            logger.error(f"Failed to replace keyword placeholder: {e}", exc_info=True)
            raise

    def _apply_source_filter(self, query_body: dict, source_filter: str) -> dict:
        """Apply source filter to the query body."""
        try:
            # Get the original query
            original_query = query_body.get("query", {})
            print("original_query------dhruv1", original_query)
            # Check if original query is already a bool query
            if "bool" in original_query: 
                print("inside the if-------")
                bool_query = original_query["bool"]
                print("Source filter, dhruv----------------------",source_filter)
                
                if source_filter == "other":
                    if "must_not" not in bool_query:
                        bool_query["must_not"] = []
                    bool_query["must_not"].append({"term": {"source": "court_listener"}})
                    bool_query["must_not"].append({"term": {"tag": "Untagged"}})
                else:
                    if "must" not in bool_query:
                        bool_query["must"] = []
                    bool_query["must"].append({"term": {"source": source_filter}})
                    bool_query["must_not"].append({"term": {"tag": "Untagged"}})
            else:
                # Wrap original query in bool
                print("adding source----", source_filter)
                if source_filter == "other":
                    print("other-------")
                    query_body["query"] = {
                        "bool": {
                            "must": [original_query],
                            "must_not": [
                                {"term": {"source": "court_listener"}},
                                {"term": {"tag": "Untagged"}}
                            ]
                            # "must_not": [{"term": {"source": "court_listener"}}],
                            # "must_not": [{"term": {"tag": "Untagged"}}]
                        }
                    }
                else:
                    query_body["query"] = {
                        "bool": {
                            "must": [
                                original_query,
                                {"term": {"source": source_filter}}
                            ],
                            "must_not": [{"term": {"tag": "Untagged"}}]
                        }
                    }
            
            return query_body
            
        except Exception as e:
            logger.error(f"Error applying source filter: {e}", exc_info=True)
            return query_body

    def _default_query(self) -> dict:
        """Return a fallback match_all query."""
        return {"query": {"match_all": {}}, "size": 500}
    
    def filter_results_by_relevance(self, results: dict, threshold: float = 0.5) -> dict:
        """
        Filter search results to only include highly relevant documents.
        
        Args:
            results: OpenSearch results dictionary with hits
            threshold: Relevance threshold (0.0-1.0). Default 0.5 = 50% of max score
            
        Returns:
            Filtered results dictionary
            
        Example:
            # Get results from OpenSearch
            results = await opensearch_client.search(query)
            
            # Filter to only get results >= 50% of max score
            filtered = generator.filter_results_by_relevance(results, threshold=0.5)
        """
        try:
            hits = results.get("hits", {}).get("hits", [])
            
            if not hits:
                logger.info("No results to filter")
                return results
            
            # Get max score from the top result
            max_score = hits[0].get("_score", 0)
            min_score_threshold = max_score * threshold
            
            logger.info(f"Max score: {max_score}, Threshold: {threshold}, Min score: {min_score_threshold}")
            
            # Filter results
            filtered_hits = [
                hit for hit in hits 
                if hit.get("_score", 0) >= min_score_threshold
            ]
            
            original_count = len(hits)
            filtered_count = len(filtered_hits)
            
            logger.info(f"Filtered results: {filtered_count}/{original_count} documents (kept {filtered_count/original_count*100:.1f}%)")
            
            # Update results
            results["hits"]["hits"] = filtered_hits
            results["hits"]["total"]["value"] = filtered_count
            
            # Add metadata about filtering
            results["_relevance_filter"] = {
                "threshold": threshold,
                "max_score": max_score,
                "min_score_threshold": min_score_threshold,
                "original_count": original_count,
                "filtered_count": filtered_count
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error filtering results by relevance: {e}", exc_info=True)
            return results
    
    def get_score_distribution(self, results: dict) -> dict:
        """
        Analyze the score distribution of search results.
        
        Args:
            results: OpenSearch results dictionary
            
        Returns:
            Dictionary with score statistics
        """
        try:
            hits = results.get("hits", {}).get("hits", [])
            
            if not hits:
                return {"error": "No results"}
            
            scores = [hit.get("_score", 0) for hit in hits]
            
            distribution = {
                "max_score": max(scores),
                "min_score": min(scores),
                "avg_score": sum(scores) / len(scores),
                "total_results": len(scores),
                "score_ranges": {
                    "high_relevance (>80% of max)": len([s for s in scores if s >= max(scores) * 0.8]),
                    "medium_relevance (50-80% of max)": len([s for s in scores if max(scores) * 0.5 <= s < max(scores) * 0.8]),
                    "low_relevance (<50% of max)": len([s for s in scores if s < max(scores) * 0.5])
                }
            }
            
            logger.info(f"Score distribution: {distribution}")
            return distribution
            
        except Exception as e:
            logger.error(f"Error calculating score distribution: {e}", exc_info=True)
            return {"error": str(e)}












# import json
# import re
# from typing import Optional, List
# from src.api.routes.bedrock_client import BedrockClient, BedrockConfig
# from src.api.routes.logger import get_logger
# from src.api.routes.concern_risk_misc_naics import concerns_events, emerging_risks, misc_topics, naics_data
# from src.api.routes.settings import BEDROCK_MODEL
# from CommonService.CommonService.async_bedrock.base import TitanV2
# import asyncio
# from CommonService.async_commonsession.commonsession import (
#             CommonSession,
#             CommonSessionConfig,
#         )

# logger = get_logger(__name__)

# # ------------------ OPENSEARCH SCHEMA ------------------

# OPENSEARCH_SCHEMA = """ 
# Index Name: ei_articles_index-05-nov-test

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

# # ------------------ PROMPT TEMPLATE ------------------

# QUERY_GENERATION_PROMPT = """
# You are an expert at converting natural language queries into valid OpenSearch DSL queries with HYBRID search (both semantic and keyword).

# INDEX SCHEMA:
# {schema}

# USER QUERY: {query}

# CRITICAL INSTRUCTIONS:
# 1. Generate ONLY a valid JSON object for the OpenSearch query body.
# 2. Use EXACT field names from the schema (case-sensitive).
# 3. ALWAYS use HYBRID search combining both semantic (KNN) and keyword search for content queries.
# 4. Query rules:
#    - For content/topic queries (about articles, topics, events):
#        → Use HYBRID search with BOTH:
#          a) KNN semantic search on `chunk_vector`
#          b) Multi-match keyword search on text fields (title, data, chunk_text, reason_identified)
#        → Use the placeholder text "__EMBEDDING_TEXT__" for semantic search
#        → Use the placeholder text "__KEYWORD_TEXT__" for keyword search
       
#    - For metadata-only searches (e.g. tag, source, region):
#        → Use only `term`, `terms`, `match`, or `range` queries on those fields.
   
#    - To combine hybrid search with filters:
#        → Use a `bool` query with `must` containing both KNN, multi_match, and filters.

# 5. Field types:
#    - Keyword: tag, source, concerns, emerging_risk_name, miscTopics, naicscode, naics_description, region, url
#    - Text (for keyword search): title, description, data, reason_identified, chunk_text
#    - Boolean: is_latest
#    - Numeric: doc_id, chunk_id
#    - Date/time: published_time, last_update_time, injection_time

# 6. Hybrid search structure (ALWAYS use this for content queries):
#    {{
#      "query": {{
#        "bool": {{
#          "must": [
#            {{
#              "knn": {{
#                "chunk_vector": {{
#                  "vector": "__EMBEDDING_TEXT__",
#                  "k": 10
#                }}
#              }}
#            }},
#            {{
#              "multi_match": {{
#                "query": "__KEYWORD_TEXT__",
#                "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified"],
#                "type": "best_fields",
#                "operator": "or"
#              }}
#            }}
#          ]
#        }}
#      }}
#    }}

# 7. Boosting strategy:
#    - title^3 (highest relevance)
#    - data^2, chunk_text^2 (medium relevance)
#    - reason_identified (base relevance)

# 8. Combination examples:
#    - "Articles about climate change":
#      → Hybrid search (KNN + multi_match) on "climate change"
   
#    - "Articles about climate change tagged as Current":
#      → bool.must with: KNN, multi_match, and term(tag="Current")
   
#    - "Show PFAS articles":
#      → Hybrid search on "PFAS" + term(concerns="PFAS")
   
#    - "Articles about wildfires in last 3 days":
#      → bool.must with: KNN, multi_match, and range on published_time

# 9. Return ONLY the JSON object — no text, markdown, or explanations.

# ---

# EXAMPLES:

# Query: "Show me all articles tagged as Current"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Current"
#     }}
#   }}
# }}

# Query: "Find articles about climate change"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "knn": {{
#             "chunk_vector": {{
#               "vector": "__EMBEDDING_TEXT__",
#               "k": 10
#             }}
#           }}
#         }},
#         {{
#           "multi_match": {{
#             "query": "__KEYWORD_TEXT__",
#             "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified"],
#             "type": "best_fields",
#             "operator": "or"
#           }}
#         }}
#       ]
#     }}
#   }}
# }}

# Query: "Find articles about lawsuits tagged as Current"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "term": {{
#             "tag": "Current"
#           }}
#         }},
#         {{
#           "knn": {{
#             "chunk_vector": {{
#               "vector": "__EMBEDDING_TEXT__",
#               "k": 10
#             }}
#           }}
#         }},
#         {{
#           "multi_match": {{
#             "query": "__KEYWORD_TEXT__",
#             "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified"],
#             "type": "best_fields",
#             "operator": "or"
#           }}
#         }}
#       ]
#     }}
#   }}
# }}

# Query: "Show untagged articles"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Untagged"
#     }}
#   }}
# }}

# Query: "Show articles about wildfire in last 3 days"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "range": {{
#             "published_time": {{
#               "gte": "now-3d/d",
#               "lte": "now"
#             }}
#           }}
#         }},
#         {{
#           "knn": {{
#             "chunk_vector": {{
#               "vector": "__EMBEDDING_TEXT__",
#               "k": 10
#             }}
#           }}
#         }},
#         {{
#           "multi_match": {{
#             "query": "__KEYWORD_TEXT__",
#             "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified"],
#             "type": "best_fields",
#             "operator": "or"
#           }}
#         }}
#       ]
#     }}
#   }}
# }}
# """



# class OpenSearchQueryGenerator:
#     def __init__(self):
#         self.bedrock = BedrockClient(BedrockConfig())
#         self.model_id = BEDROCK_MODEL
#         self.embedding_model_id = "amazon.titan-embed-text-v2:0"

#     def _prepare_schema(self) -> str:
#         """Prepare schema for the prompt."""
#         return OPENSEARCH_SCHEMA.format(
#             concerns=", ".join(concerns_events) + "...",
#             emerging_risks=", ".join(emerging_risks) + "...",
#             misc_topics=", ".join(misc_topics) + "...",
#             naics_data=", ".join([f"{item['code']}" for item in naics_data]) + "..."
#         )
    
#     async def generate_embeddings(self, text: str) -> List[float]:
#         """
#         Generate embedding vector for given text using Amazon Titan Embedding model.
        
#         Args:
#             text: The text to generate embeddings for
            
#         Returns:
#             List of floats representing the embedding vector (1024 dimensions for Titan v2)
#         """
#         try:
#             logger.info(f"Generating embedding for text: {text[:100]}...")
            
#             async with CommonSession(
#                 CommonSessionConfig(
#                     client_name="bedrock-runtime",
#                     region="us-east-1",
#                     profile_name="Comm-Prop-Sandbox"
#                 )
#             ) as titan:
#                 titan_embedding = TitanV2(titan)
                
#                 # ✅ CRITICAL: Pass dict, not JSON string
#                 # invoke_with_retry will call json.dumps() internally (base.py line 70)
#                 payload = {"inputText": text}
#                 response = await titan_embedding.generate_embedding(payload)
                
#                 # TitanV2.generate_embedding returns the embedding vector directly
#                 # as a list of floats (see base.py line 156 and line 72)
#                 if not isinstance(response, list):
#                     logger.error(f"Unexpected response type: {type(response)}, value: {response}")
#                     raise ValueError(f"Expected list of floats, got {type(response)}")
                
#                 if len(response) == 0:
#                     raise ValueError("Received empty embedding vector")
                
#                 logger.info(f"✅ Successfully generated {len(response)}-dimensional embedding")
#                 return response
                
#         except Exception as e:
#             logger.error(f"Failed to generate embedding: {e}", exc_info=True)
#             raise
    
#     async def _create_hybrid_query(self, user_query: str):
#         """
#         Build a hybrid query combining both semantic (KNN) and keyword (multi_match) search.
#         """
#         embedding_vector = await self.generate_embeddings(user_query)
#         logger.info(f"Generated embedding vector type: {type(embedding_vector)}, length: {len(embedding_vector)}")
        
#         # ✅ Hybrid search: KNN + multi_match
#         return {
#             "size": 50,
#             "query": {
#                 "bool": {
#                     "must": [
#                         {
#                             "knn": {
#                                 "chunk_vector": {
#                                     "vector": embedding_vector,
#                                     "k": 10
#                                 }
#                             }
#                         },
#                         {
#                             "multi_match": {
#                                 "query": user_query,
#                                 "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified"],
#                                 "type": "best_fields",
#                                 "operator": "or"
#                             }
#                         }
#                     ]
#                 }
#             }
#         }

#     def _needs_semantic_search(self, user_query: str) -> bool:
#         """
#         Determine whether the user query should trigger a semantic (vector) search.
#         Returns True if the query likely targets 'data', 'reason_identified', or 'title'
#         fields, or contains general article-topic language.
#         """
#         if not user_query:
#             return False

#         query_lower = user_query.lower()
        
#         # Metadata-only keywords that don't need semantic search
#         metadata_only_keywords = [
#             "show all", "tagged as", "from source", "in region",
#             "with tag", "untagged", "current", "potential new trend"
#         ]
        
#         # Check if query is purely metadata-based
#         for keyword in metadata_only_keywords:
#             if keyword in query_lower and "about" not in query_lower and "related" not in query_lower:
#                 return False
        
#         # Semantic search keywords
#         semantic_keywords = [
#             "about", "related to", "discuss", "concerning",
#             "impact", "risk", "article", "topic", "find",
#             "search for", "looking for", "show me articles on"
#         ]

#         # If any semantic keyword appears, we need semantic search
#         return any(keyword in query_lower for keyword in semantic_keywords)

#     async def _add_hybrid_query(self, user_query: str, k: int = 10):
#         """
#         Add hybrid search (KNN + multi_match) if semantic search is required but missing.
#         """
#         logger.info("No hybrid search found but content search needed, adding hybrid query")

#         embedding_vector = await self.generate_embeddings(user_query)
#         if not isinstance(embedding_vector, list):
#             raise ValueError("Embedding vector must be list of floats")

#         hybrid_query = {
#             "bool": {
#                 "must": [
#                     {
#                         "knn": {
#                             "chunk_vector": {
#                                 "vector": embedding_vector,
#                                 "k": k
#                             }
#                         }
#                     },
#                     {
#                         "multi_match": {
#                             "query": user_query,
#                             "fields": ["title^3", "data^2", "chunk_text^2", "reason_identified"],
#                             "type": "best_fields",
#                             "operator": "or"
#                         }
#                     }
#                 ]
#             }
#         }

#         logger.info(f"✅ Added fallback hybrid query for: {user_query}")
#         return {"query": hybrid_query, "size": 50}

#     async def generate_query(self, user_query: str, source_filter: Optional[str] = None) -> dict:
#         """Generate an OpenSearch query DSL body from a natural language query."""
#         try:
#             schema = self._prepare_schema()
#             prompt = QUERY_GENERATION_PROMPT.format(schema=schema, query=user_query)

#             # Get LLM to generate query structure
#             response_text = self.bedrock.invoke_model(
#                 model_id=self.model_id,
#                 prompt=prompt,
#                 max_tokens=2000,
#                 temperature=0.0
#             )

#             logger.info(f"=== LLM FULL Response ===\n{response_text}\n=== END Response ===")
            
#             query_body = self._extract_json(response_text)
#             logger.info(f"=== Extracted query_body ===\n{json.dumps(query_body, indent=2)}\n=== END query_body ===")

#             # Check if we need semantic search
#             needs_semantic_search = self._needs_semantic_search(user_query)
#             logger.info(f"User query needs semantic/content search: {needs_semantic_search}")

#             if not query_body or ("query" not in query_body and "knn" not in query_body):
#                 logger.warning(f"Invalid query generated. Response: {response_text[:200]}")
#                 if needs_semantic_search:
#                     logger.info("Generating hybrid query directly as fallback")
#                     query_body = await self._create_hybrid_query(user_query)
#                 else:
#                     query_body = self._default_query()

#             # Process placeholders - find and replace __EMBEDDING_TEXT__ and __KEYWORD_TEXT__
#             try:
#                 processed = await self._process_placeholders(query_body, user_query)
#                 logger.info(f"Placeholders processed: {processed}")
                
#                 # If no hybrid search was found but query needs it, add it
#                 if not processed and needs_semantic_search:
#                     logger.info("No hybrid search found but content search needed, adding hybrid query")
#                     query_body = await self._create_hybrid_query(user_query)
                    
#             except Exception as e:
#                 logger.error(f"Failed to process placeholders: {e}")
#                 if needs_semantic_search:
#                     logger.info("Falling back to direct hybrid query creation")
#                     query_body = await self._create_hybrid_query(user_query)
#                 else:
#                     return self._default_query()

#             # Ensure size parameter is set
#             if "size" not in query_body:
#                 query_body["size"] = 50

#             # Add source filter if provided
#             if source_filter:
#                 query_body = self._apply_source_filter(query_body, source_filter)

#             logger.info(f"Successfully generated OpenSearch query for: {user_query}")
#             if source_filter:
#                 logger.info(f"Source filter applied: {source_filter}")
#             logger.info(f"=== FINAL Query body ===\n{json.dumps(query_body, indent=2)}\n=== END FINAL ===")
#             return query_body

#         except Exception as e:
#             logger.error(f"Error generating OpenSearch query: {e}", exc_info=True)
#             return self._default_query()

#     def _extract_json(self, text: str) -> dict:
#         """Extract JSON from the LLM's response text with robust handling."""
#         text = text.strip()
#         logger.debug(f"Raw LLM output (first 500 chars): {text[:500]}")

#         # Common cleanup for markdown/code fences
#         text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
#         text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
#         text = text.strip("` \n\t")

#         # Try to find JSON object anywhere in text
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if not match:
#             logger.error("No JSON braces found in response.")
#             return {}

#         candidate = match.group(0)
#         try:
#             parsed = json.loads(candidate)
#             return parsed
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to decode JSON: {e}. Candidate snippet: {candidate[:300]}")
#             return {}

#     async def _process_placeholders(self, query_body: dict, user_query: str) -> bool:
#         """
#         Recursively find and replace both __EMBEDDING_TEXT__ and __KEYWORD_TEXT__ placeholders.
        
#         Returns:
#             bool: True if at least one placeholder was found and processed
#         """
#         placeholder_found = False
        
#         async def process_dict(obj, parent_key=None):
#             nonlocal placeholder_found
            
#             if isinstance(obj, dict):
#                 # Process KNN queries with __EMBEDDING_TEXT__
#                 if "knn" in obj:
#                     knn_query = obj["knn"]
#                     if isinstance(knn_query, dict):
#                         await self._replace_knn_placeholder(knn_query, user_query)
#                         placeholder_found = True
                
#                 # Process multi_match queries with __KEYWORD_TEXT__
#                 if "multi_match" in obj:
#                     multi_match_query = obj["multi_match"]
#                     if isinstance(multi_match_query, dict):
#                         self._replace_keyword_placeholder(multi_match_query, user_query)
#                         placeholder_found = True
                
#                 # Recursively process all nested structures
#                 for key, value in obj.items():
#                     if isinstance(value, (dict, list)):
#                         await process_dict(value, key)
            
#             elif isinstance(obj, list):
#                 for item in obj:
#                     if isinstance(item, (dict, list)):
#                         await process_dict(item, parent_key)
        
#         await process_dict(query_body)
#         return placeholder_found

#     async def _replace_knn_placeholder(self, knn_query: dict, user_query: str) -> None:
#         """
#         Replace __EMBEDDING_TEXT__ placeholder in KNN query with actual embedding vector.
#         """
#         try:
#             # KNN query structure: {"chunk_vector": {"vector": "...", "k": 10}}
#             for field_name, field_config in knn_query.items():
#                 if isinstance(field_config, dict) and "vector" in field_config:
#                     vector_value = field_config["vector"]
                    
#                     # Check if it's a placeholder or text that needs embedding
#                     if isinstance(vector_value, str):
#                         if "__EMBEDDING_TEXT__" in vector_value:
#                             embedding_text = user_query
#                             logger.info(f"Found __EMBEDDING_TEXT__ placeholder, using user query: {user_query}")
#                         else:
#                             embedding_text = vector_value
#                             logger.info(f"Found text in vector field: {embedding_text}")
                        
#                         # Generate the actual embedding vector
#                         embedding_vector = await self.generate_embeddings(embedding_text)
                        
#                         # Replace with actual vector
#                         field_config["vector"] = embedding_vector
                        
#                         logger.info(f"✅ Successfully replaced KNN placeholder with {len(embedding_vector)}-dim embedding vector")
#                     elif isinstance(vector_value, list):
#                         # Already has a vector, skip
#                         logger.info(f"Field {field_name} already has embedding vector")
#                         continue
            
#         except Exception as e:
#             logger.error(f"Failed to replace KNN placeholder: {e}", exc_info=True)
#             raise

#     def _replace_keyword_placeholder(self, multi_match_query: dict, user_query: str) -> None:
#         """
#         Replace __KEYWORD_TEXT__ placeholder in multi_match query with actual search text.
#         """
#         try:
#             if "query" in multi_match_query:
#                 query_value = multi_match_query["query"]
                
#                 if isinstance(query_value, str) and "__KEYWORD_TEXT__" in query_value:
#                     multi_match_query["query"] = user_query
#                     logger.info(f"✅ Successfully replaced __KEYWORD_TEXT__ with: {user_query}")
#                 elif isinstance(query_value, str):
#                     # Already has text, might be fine
#                     logger.info(f"multi_match already has query text: {query_value}")
            
#         except Exception as e:
#             logger.error(f"Failed to replace keyword placeholder: {e}", exc_info=True)
#             raise

#     def _apply_source_filter(self, query_body: dict, source_filter: str) -> dict:
#         """Apply source filter to the query body."""
#         try:
#             # Get the original query
#             original_query = query_body.get("query", {})
            
#             # Check if original query is already a bool query
#             if "bool" in original_query:
#                 bool_query = original_query["bool"]
                
#                 if source_filter == "other":
#                     if "must_not" not in bool_query:
#                         bool_query["must_not"] = []
#                     bool_query["must_not"].append({"term": {"source": "court_listener"}})
#                 else:
#                     if "must" not in bool_query:
#                         bool_query["must"] = []
#                     bool_query["must"].append({"term": {"source": source_filter}})
#             else:
#                 # Wrap original query in bool
#                 if source_filter == "other":
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [original_query],
#                             "must_not": [{"term": {"source": "court_listener"}}]
#                         }
#                     }
#                 else:
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [
#                                 original_query,
#                                 {"term": {"source": source_filter}}
#                             ]
#                         }
#                     }
            
#             return query_body
            
#         except Exception as e:
#             logger.error(f"Error applying source filter: {e}", exc_info=True)
#             return query_body

#     def _default_query(self) -> dict:
#         """Return a fallback match_all query."""
#         return {"query": {"match_all": {}}, "size": 50}


















# import json
# import re
# from typing import Optional, List
# from src.api.routes.bedrock_client import BedrockClient, BedrockConfig
# from src.api.routes.logger import get_logger
# from src.api.routes.concern_risk_misc_naics import concerns_events, emerging_risks, misc_topics, naics_data
# from src.api.routes.settings import BEDROCK_MODEL
# from CommonService.CommonService.async_bedrock.base import TitanV2
# import asyncio
# from CommonService.async_commonsession.commonsession import (
#             CommonSession,
#             CommonSessionConfig,
#         )

# logger = get_logger(__name__)

# # ------------------ OPENSEARCH SCHEMA ------------------

# OPENSEARCH_SCHEMA = """ 
# Index Name: ei_articles_index-05-nov-test

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

# # ------------------ PROMPT TEMPLATE ------------------

# QUERY_GENERATION_PROMPT = """
# You are an expert at converting natural language queries into valid OpenSearch DSL queries.

# INDEX SCHEMA:
# {schema}

# USER QUERY: {query}

# CRITICAL INSTRUCTIONS:
# 1. Generate ONLY a valid JSON object for the OpenSearch query body.
# 2. Use EXACT field names from the schema (case-sensitive).
# 3. Query rules:
#    - If the query involves *content*, *title*, *reason*, or any semantic topic (e.g. mentions data, reason_identified, title, or general article topic):
#        → Use a KNN query on `chunk_vector`.
#        → Use the placeholder text "__EMBEDDING_TEXT__" for the semantic search text
#        Example:
#        {{
#          "query": {{
#            "knn": {{
#              "chunk_vector": {{
#                "vector": "__EMBEDDING_TEXT__",
#                "k": 10
#              }}
#            }}
#          }}
#        }}
#    - For metadata searches (e.g. tag, source, region, concerns, emerging_risk_name, miscTopics, naicscode):
#        → Use `term`, `terms`, `match`, or `range` queries on those fields.
#    - To combine similarity and filters (hybrid search):
#        → Use a `bool` query where `must` includes both the `knn` and any filters.
#        ✅ The `knn` query should be placed directly in `must` array.

# 4. Field types:
#    - Keyword: tag, source, concerns, emerging_risk_name, miscTopics, naicscode, naics_description, region, url
#    - Text: title, description, data, reason_identified, chunk_text
#    - Boolean: is_latest
#    - Numeric: doc_id, chunk_id
#    - Date/time: published_time, last_update_time, injection_time

# 5. Common query patterns:
#    - "Show all articles" → match_all
#    - "Tagged as Current" → term query on tag
#    - "From Reuters" → term query on source
#    - "With concern PFAS" → term query on concerns
#    - "Show recent articles" → range query on published_time or last_update_time

# 6. Combination examples:
#    - "Articles about climate change tagged as Current":
#      → Use `bool.must` with knn query for "climate change" and a term query for tag="Current".
#    - "Emerging risks in Europe":
#      → term on region="Europe" + knn if context indicates semantic topic.
#    - "Show PFAS untagged articles":
#      → bool.must with term(tag="Untagged") and term(concerns="PFAS").
#    - "Articles about wildfires in last 3 days":
#      → bool.must with range on published_time and knn for "wildfires".

# 7. Always structure hybrid queries like this:
#    {{
#      "query": {{
#        "bool": {{
#          "must": [
#            <other field filters>,
#            {{
#              "knn": {{
#                "chunk_vector": {{
#                  "vector": "__EMBEDDING_TEXT__",
#                  "k": 10
#                }}
#              }}
#            }}
#          ]
#        }}
#      }}
#    }}

# 8. Return ONLY the JSON object — no text, markdown, or explanations.

# ---

# EXAMPLES:

# Query: "Show me all articles tagged as Current"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Current"
#     }}
#   }}
# }}

# Query: "Find articles about climate change"
# {{
#   "query": {{
#     "knn": {{
#       "chunk_vector": {{
#         "vector": "__EMBEDDING_TEXT__",
#         "k": 10
#       }}
#     }}
#   }}
# }}

# Query: "Find articles about lawsuits tagged as Current"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "term": {{
#             "tag": "Current"
#           }}
#         }},
#         {{
#           "knn": {{
#             "chunk_vector": {{
#               "vector": "__EMBEDDING_TEXT__",
#               "k": 10
#             }}
#           }}
#         }}
#       ]
#     }}
#   }}
# }}

# Query: "Show untagged articles"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Untagged"
#     }}
#   }}
# }}

# Query: "Show articles with wildfire for last 3 days"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "range": {{
#             "published_time": {{
#               "gte": "now-3d/d",
#               "lte": "now"
#             }}
#           }}
#         }},
#         {{
#           "knn": {{
#             "chunk_vector": {{
#               "vector": "__EMBEDDING_TEXT__",
#               "k": 10
#             }}
#           }}
#         }}
#       ]
#     }}
#   }}
# }}
# """



# class OpenSearchQueryGenerator:
#     def __init__(self):
#         self.bedrock = BedrockClient(BedrockConfig())
#         self.model_id = BEDROCK_MODEL
#         self.embedding_model_id = "amazon.titan-embed-text-v2:0"

#     def _prepare_schema(self) -> str:
#         """Prepare schema for the prompt."""
#         return OPENSEARCH_SCHEMA.format(
#             concerns=", ".join(concerns_events) + "...",
#             emerging_risks=", ".join(emerging_risks) + "...",
#             misc_topics=", ".join(misc_topics) + "...",
#             naics_data=", ".join([f"{item['code']}" for item in naics_data]) + "..."
#         )
    
#     async def generate_embeddings(self, text: str) -> List[float]:
#         """
#         Generate embedding vector for given text using Amazon Titan Embedding model.
        
#         Args:
#             text: The text to generate embeddings for
            
#         Returns:
#             List of floats representing the embedding vector (1024 dimensions for Titan v2)
#         """
#         try:
#             logger.info(f"Generating embedding for text: {text[:100]}...")
            
#             async with CommonSession(
#                 CommonSessionConfig(
#                     client_name="bedrock-runtime",
#                     region="us-east-1",
#                     profile_name="Comm-Prop-Sandbox"
#                 )
#             ) as titan:
#                 titan_embedding = TitanV2(titan)
                
#                 # ✅ CRITICAL: Pass dict, not JSON string
#                 # invoke_with_retry will call json.dumps() internally (base.py line 70)
#                 payload = {"inputText": text}
#                 response = await titan_embedding.generate_embedding(payload)
                
#                 # TitanV2.generate_embedding returns the embedding vector directly
#                 # as a list of floats (see base.py line 156 and line 72)
#                 if not isinstance(response, list):
#                     logger.error(f"Unexpected response type: {type(response)}, value: {response}")
#                     raise ValueError(f"Expected list of floats, got {type(response)}")
                
#                 if len(response) == 0:
#                     raise ValueError("Received empty embedding vector")
                
#                 logger.info(f"✅ Successfully generated {len(response)}-dimensional embedding")
#                 return response
                
#         except Exception as e:
#             logger.error(f"Failed to generate embedding: {e}", exc_info=True)
#             raise
    
#     async def _create_knn_query(self, user_query: str):
#         """
#         Build a proper KNN query for OpenSearch using the 1024-dim embedding.
#         """
#         embedding_vector = await self.generate_embeddings(user_query)
#         logger.info(f"Generated embedding vector type: {type(embedding_vector)}, length: {len(embedding_vector)}")
        
#         # ✅ CORRECT: Use the proper OpenSearch KNN query structure
#         return {
#             "size": 50,
#             "query": {
#                 "knn": {
#                     "chunk_vector": {
#                         "vector": embedding_vector,
#                         "k": 10
#                     }
#                 }
#             }
#         }

#     def _needs_semantic_search(self, user_query: str) -> bool:
#         """
#         Determine whether the user query should trigger a semantic (vector) search.
#         Returns True if the query likely targets 'data', 'reason_identified', or 'title'
#         fields, or contains general article-topic language.
#         """
#         if not user_query:
#             return False

#         query_lower = user_query.lower()
#         semantic_keywords = [
#             "data", "reason_identified", "title",
#             "about", "related to", "discuss", "concern",
#             "impact", "risk", "article", "topic", "find"
#         ]

#         # If any keyword appears, we assume semantic search is needed
#         return any(keyword in query_lower for keyword in semantic_keywords)

#     async def _add_knn_to_query(self, user_query: str, k: int = 10):
#         """
#         Add KNN block dynamically if semantic search is required but missing.
#         """
#         logger.info("No KNN found but semantic search needed, adding KNN query")

#         embedding_vector = await self.generate_embeddings(user_query)
#         if not isinstance(embedding_vector, list):
#             raise ValueError("Embedding vector must be list of floats")

#         knn_query = {
#             "knn": {
#                 "chunk_vector": {
#                     "vector": embedding_vector,
#                     "k": k
#                 }
#             }
#         }

#         logger.info(f"✅ Added fallback KNN query for: {user_query}")
#         return knn_query

#     async def generate_query(self, user_query: str, source_filter: Optional[str] = None) -> dict:
#         """Generate an OpenSearch query DSL body from a natural language query."""
#         try:
#             schema = self._prepare_schema()
#             prompt = QUERY_GENERATION_PROMPT.format(schema=schema, query=user_query)

#             # Get LLM to generate query structure
#             response_text = self.bedrock.invoke_model(
#                 model_id=self.model_id,
#                 prompt=prompt,
#                 max_tokens=2000,
#                 temperature=0.0
#             )

#             logger.info(f"=== LLM FULL Response ===\n{response_text}\n=== END Response ===")
            
#             query_body = self._extract_json(response_text)
#             logger.info(f"=== Extracted query_body ===\n{json.dumps(query_body, indent=2)}\n=== END query_body ===")

#             # Check if we need semantic search
#             needs_semantic_search = self._needs_semantic_search(user_query)
#             logger.info(f"User query needs semantic search: {needs_semantic_search}")

#             if not query_body or ("query" not in query_body and "knn" not in query_body):
#                 logger.warning(f"Invalid query generated. Response: {response_text[:200]}")
#                 if needs_semantic_search:
#                     logger.info("Generating KNN query directly as fallback")
#                     query_body = await self._create_knn_query(user_query)
#                 else:
#                     query_body = self._default_query()

#             # Process KNN queries - find placeholders and replace with actual embeddings
#             try:
#                 knn_processed = await self._process_knn_embeddings(query_body, user_query)
#                 logger.info(f"KNN embeddings processed: {knn_processed}")
                
#                 # If no KNN was found but query needs semantic search, add it
#                 if not knn_processed and needs_semantic_search:
#                     logger.info("No KNN found but semantic search needed, adding KNN query")
#                     query_body = await self._create_knn_query(user_query)
                    
#             except Exception as e:
#                 logger.error(f"Failed to process KNN embeddings: {e}")
#                 if needs_semantic_search:
#                     logger.info("Falling back to direct KNN query creation")
#                     query_body = await self._create_knn_query(user_query)
#                 else:
#                     return self._default_query()

#             # Ensure size parameter is set
#             if "size" not in query_body:
#                 query_body["size"] = 50

#             # Add source filter if provided
#             if source_filter:
#                 query_body = self._apply_source_filter(query_body, source_filter)

#             logger.info(f"Successfully generated OpenSearch query for: {user_query}")
#             if source_filter:
#                 logger.info(f"Source filter applied: {source_filter}")
#             logger.info(f"=== FINAL Query body ===\n{json.dumps(query_body, indent=2)}\n=== END FINAL ===")
#             return query_body

#         except Exception as e:
#             logger.error(f"Error generating OpenSearch query: {e}", exc_info=True)
#             return self._default_query()

#     def _extract_json(self, text: str) -> dict:
#         """Extract JSON from the LLM's response text with robust handling."""
#         text = text.strip()
#         logger.debug(f"Raw LLM output (first 500 chars): {text[:500]}")

#         # Common cleanup for markdown/code fences
#         text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
#         text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
#         text = text.strip("` \n\t")

#         # Try to find JSON object anywhere in text
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if not match:
#             logger.error("No JSON braces found in response.")
#             return {}

#         candidate = match.group(0)
#         try:
#             parsed = json.loads(candidate)
#             return parsed
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to decode JSON: {e}. Candidate snippet: {candidate[:300]}")
#             return {}

#     async def _process_knn_embeddings(self, query_body: dict, user_query: str) -> bool:
#         """
#         Recursively find KNN queries with placeholder text and replace with actual embeddings.
#         Handles KNN queries at any nesting level (root, bool.must, bool.should, etc.)
        
#         Returns:
#             bool: True if at least one KNN query was found and processed
#         """
#         knn_found = False
        
#         async def process_dict(obj, parent_key=None):
#             nonlocal knn_found
#             if isinstance(obj, dict):
#                 # Check if this dict contains a KNN query with placeholder
#                 if "knn" in obj:
#                     knn_query = obj["knn"]
#                     if isinstance(knn_query, dict):
#                         await self._replace_placeholder_with_embedding(knn_query, user_query)
#                         knn_found = True
                
#                 # Recursively process all nested structures
#                 for key, value in obj.items():
#                     if isinstance(value, (dict, list)):
#                         await process_dict(value, key)
            
#             elif isinstance(obj, list):
#                 for item in obj:
#                     if isinstance(item, (dict, list)):
#                         await process_dict(item, parent_key)
        
#         await process_dict(query_body)
#         return knn_found

#     async def _replace_placeholder_with_embedding(self, knn_query: dict, user_query: str) -> None:
#         """
#         Replace placeholder text in KNN query with actual embedding vector.
#         Supports the chunk_vector field structure used in OpenSearch.
#         """
#         try:
#             # KNN query structure: {"chunk_vector": {"vector": "...", "k": 10}}
#             for field_name, field_config in knn_query.items():
#                 if isinstance(field_config, dict) and "vector" in field_config:
#                     vector_value = field_config["vector"]
                    
#                     # Check if it's a placeholder or text that needs embedding
#                     if isinstance(vector_value, str):
#                         if "__EMBEDDING_TEXT__" in vector_value:
#                             embedding_text = user_query
#                             logger.info(f"Found placeholder, using user query: {user_query}")
#                         else:
#                             embedding_text = vector_value
#                             logger.info(f"Found text in vector field: {embedding_text}")
                        
#                         # Generate the actual embedding vector
#                         embedding_vector = await self.generate_embeddings(embedding_text)
                        
#                         # Replace with actual vector
#                         field_config["vector"] = embedding_vector
                        
#                         logger.info(f"✅ Successfully replaced placeholder with {len(embedding_vector)}-dim embedding vector")
#                     elif isinstance(vector_value, list):
#                         # Already has a vector, skip
#                         logger.info(f"Field {field_name} already has embedding vector")
#                         continue
            
#         except Exception as e:
#             logger.error(f"Failed to replace placeholder with embedding: {e}", exc_info=True)
#             raise

#     def _apply_source_filter(self, query_body: dict, source_filter: str) -> dict:
#         """Apply source filter to the query body."""
#         try:
#             # Get the original query
#             original_query = query_body.get("query", {})
            
#             # Check if original query is already a bool query
#             if "bool" in original_query:
#                 bool_query = original_query["bool"]
                
#                 if source_filter == "other":
#                     if "must_not" not in bool_query:
#                         bool_query["must_not"] = []
#                     bool_query["must_not"].append({"term": {"source": "court_listener"}})
#                 else:
#                     if "must" not in bool_query:
#                         bool_query["must"] = []
#                     bool_query["must"].append({"term": {"source": source_filter}})
#             else:
#                 # Wrap original query in bool
#                 if source_filter == "other":
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [original_query],
#                             "must_not": [{"term": {"source": "court_listener"}}]
#                         }
#                     }
#                 else:
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [
#                                 original_query,
#                                 {"term": {"source": source_filter}}
#                             ]
#                         }
#                     }
            
#             return query_body
            
#         except Exception as e:
#             logger.error(f"Error applying source filter: {e}", exc_info=True)
#             return query_body

#     def _default_query(self) -> dict:
#         """Return a fallback match_all query."""
#         return {"query": {"match_all": {}}, "size": 50}















# import json
# import re
# from typing import Optional, List
# from src.api.routes.bedrock_client import BedrockClient, BedrockConfig
# from src.api.routes.logger import get_logger
# from src.api.routes.concern_risk_misc_naics import concerns_events, emerging_risks, misc_topics, naics_data
# from src.api.routes.settings import BEDROCK_MODEL
# from CommonService.CommonService.async_bedrock.base import TitanV2
# import asyncio
# from CommonService.async_commonsession.commonsession import (
#             CommonSession,
#             CommonSessionConfig,
#         )

# logger = get_logger(__name__)

# # ------------------ OPENSEARCH SCHEMA ------------------

# OPENSEARCH_SCHEMA = """ 
# Index Name: ei_articles_index-05-nov-test

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

# # ------------------ PROMPT TEMPLATE ------------------

# QUERY_GENERATION_PROMPT = """
# You are an expert at converting natural language queries into valid OpenSearch DSL queries.

# INDEX SCHEMA:
# {schema}

# USER QUERY: {query}

# CRITICAL INSTRUCTIONS:
# 1. Generate ONLY a valid JSON object for the OpenSearch query body.
# 2. Use EXACT field names from the schema (case-sensitive).
# 3. Query rules:
#    - If the query involves *content*, *title*, *reason*, or any semantic topic (e.g. mentions data, reason_identified, title, or general article topic):
#        → Use a similarity (vector) search on `chunk_vector` with a KNN query.
#        → Use the placeholder text "__EMBEDDING_TEXT__" for the semantic search text
#        Example:
#        {{
#          "knn": {{
#            "field": "chunk_vector",
#            "query_vector": "__EMBEDDING_TEXT__",
#            "k": 10,
#            "num_candidates": 100
#          }}
#        }}
#    - For metadata searches (e.g. tag, source, region, concerns, emerging_risk_name, miscTopics, naicscode):
#        → Use `term`, `terms`, `match`, or `range` queries on those fields.
#    - To combine similarity and filters (hybrid search):
#        → Use a `bool` query where `must` includes both the `knn` and any filters.
#        ✅ The `knn` query should be placed directly in `must` array.

# 4. Field types:
#    - Keyword: tag, source, concerns, emerging_risk_name, miscTopics, naicscode, naics_description, region, url
#    - Text: title, description, data, reason_identified, chunk_text
#    - Boolean: is_latest
#    - Numeric: doc_id, chunk_id
#    - Date/time: published_time, last_update_time, injection_time

# 5. Common query patterns:
#    - "Show all articles" → match_all
#    - "Tagged as Current" → term query on tag
#    - "From Reuters" → term query on source
#    - "With concern PFAS" → term query on concerns
#    - "Show recent articles" → range query on published_time or last_update_time

# 6. Combination examples:
#    - "Articles about climate change tagged as Current":
#      → Use `bool.must` with knn query for "climate change" and a term query for tag="Current".
#    - "Emerging risks in Europe":
#      → term on region="Europe" + knn if context indicates semantic topic.
#    - "Show PFAS untagged articles":
#      → bool.must with term(tag="Untagged") and term(concerns="PFAS").
#    - "Articles about wildfires in last 3 days":
#      → bool.must with range on published_time and knn for "wildfires".

# 7. Always structure hybrid queries like this:
#    {{
#      "query": {{
#        "bool": {{
#          "must": [
#            <other field filters>,
#            {{
#              "knn": {{
#                "field": "chunk_vector",
#                "query_vector": "__EMBEDDING_TEXT__",
#                "k": 10,
#                "num_candidates": 100
#              }}
#            }}
#          ]
#        }}
#      }}
#    }}

# 8. Return ONLY the JSON object — no text, markdown, or explanations.

# ---

# EXAMPLES:

# Query: "Show me all articles tagged as Current"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Current"
#     }}
#   }}
# }}

# Query: "Find articles about climate change"
# {{
#   "knn": {{
#     "field": "chunk_vector",
#     "query_vector": "__EMBEDDING_TEXT__",
#     "k": 10,
#     "num_candidates": 100
#   }}
# }}

# Query: "Find articles about lawsuits tagged as Current"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "term": {{
#             "tag": "Current"
#           }}
#         }},
#         {{
#           "knn": {{
#             "field": "chunk_vector",
#             "query_vector": "__EMBEDDING_TEXT__",
#             "k": 10,
#             "num_candidates": 100
#           }}
#         }}
#       ]
#     }}
#   }}
# }}

# Query: "Show untagged articles"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Untagged"
#     }}
#   }}
# }}

# Query: "Show articles with wildfire for last 3 days"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "range": {{
#             "published_time": {{
#               "gte": "now-3d/d",
#               "lte": "now"
#             }}
#           }}
#         }},
#         {{
#           "knn": {{
#             "field": "chunk_vector",
#             "query_vector": "__EMBEDDING_TEXT__",
#             "k": 10,
#             "num_candidates": 100
#           }}
#         }}
#       ]
#     }}
#   }}
# }}
# """



# class OpenSearchQueryGenerator:
#     def __init__(self):
#         self.bedrock = BedrockClient(BedrockConfig())
#         self.model_id = BEDROCK_MODEL
#         self.embedding_model_id = "amazon.titan-embed-text-v2:0"

#     def _prepare_schema(self) -> str:
#         """Prepare schema for the prompt."""
#         return OPENSEARCH_SCHEMA.format(
#             concerns=", ".join(concerns_events) + "...",
#             emerging_risks=", ".join(emerging_risks) + "...",
#             misc_topics=", ".join(misc_topics) + "...",
#             naics_data=", ".join([f"{item['code']}" for item in naics_data]) + "..."
#         )
    
#     async def generate_embeddings(self, text: str) -> List[float]:
#         """
#         Generate embedding vector for given text using Amazon Titan Embedding model.
        
#         Args:
#             text: The text to generate embeddings for
            
#         Returns:
#             List of floats representing the embedding vector (1024 dimensions for Titan v2)
#         """
#         try:
#             logger.info(f"Generating embedding for text: {text[:100]}...")
            
#             async with CommonSession(
#                 CommonSessionConfig(
#                     client_name="bedrock-runtime",
#                     region="us-east-1",
#                     profile_name="Comm-Prop-Sandbox"
#                 )
#             ) as titan:
#                 titan_embedding = TitanV2(titan)
                
#                 # ✅ CRITICAL: Pass dict, not JSON string
#                 # invoke_with_retry will call json.dumps() internally (base.py line 70)
#                 payload = {"inputText": text}
#                 response = await titan_embedding.generate_embedding(payload)
                
#                 # TitanV2.generate_embedding returns the embedding vector directly
#                 # as a list of floats (see base.py line 156 and line 72)
#                 if not isinstance(response, list):
#                     logger.error(f"Unexpected response type: {type(response)}, value: {response}")
#                     raise ValueError(f"Expected list of floats, got {type(response)}")
                
#                 if len(response) == 0:
#                     raise ValueError("Received empty embedding vector")
                
#                 logger.info(f"✅ Successfully generated {len(response)}-dimensional embedding")
#                 return response
                
#         except Exception as e:
#             logger.error(f"Failed to generate embedding: {e}", exc_info=True)
#             raise
    
#     async def _create_knn_query(self, user_query: str):
#         """
#         Build a proper KNN query for OpenSearch using the 1024-dim embedding.
#         """
#         embedding_vector = await self.generate_embeddings(user_query)
#         logger.info(f"Generated embedding vector type: {type(embedding_vector)}, length: {len(embedding_vector)}")
        

#         return {
#             "size": 50,
#             "query": {
#                 "match": {
#                     "chunk_vector": {
#                         "vector": embedding_vector,
#                         "k": 10
#                     }
#                 }
#             }
#         }
#         # return {
#         #     "size": 50,
#         #     "knn": {
#         #         "chunk_vector": {
#         #             "vector": embedding_vector,
#         #             "k": 10,
#         #             "num_candidates": 100
#         #         }
#         #     }
#         # }
        
        
#         # return {
#         #     "knn": {
#         #         "field": "chunk_vector",
#         #         "query_vector": embedding_vector,
#         #         "k": 10,
#         #         "num_candidates": 100
#         #     }
#         # }
#         # return {
#         #   "size": 50,
#         #   "query": {
#         #       "knn": {
#         #           "field": "chunk_vector",
#         #           "query_vector": embedding_vector,
#         #           "k": 10,
#         #           "num_candidates": 100
#         #       }
#         #   }
#       # }

#         # return {
#         #     "query": {
#         #         "knn": {
#         #             "chunk_vector": {
#         #                 "vector": embedding_vector,
#         #                 "k": 10,
#         #                 "num_candidates": 100
#         #             }
#         #         }
#         #     },
#         #     "size": 50
#         # }


#     def _needs_semantic_search(self, user_query: str) -> bool:
#         """
#         Determine whether the user query should trigger a semantic (vector) search.
#         Returns True if the query likely targets 'data', 'reason_identified', or 'title'
#         fields, or contains general article-topic language.
#         """
#         if not user_query:
#             return False

#         query_lower = user_query.lower()
#         semantic_keywords = [
#             "data", "reason_identified", "title",
#             "about", "related to", "discuss", "concern",
#             "impact", "risk", "article", "topic"
#         ]

#         # If any keyword appears, we assume semantic search is needed
#         return any(keyword in query_lower for keyword in semantic_keywords)

#     async def _add_knn_to_query(self, user_query: str, k: int = 10, num_candidates: int = 100):
#         """
#         Add KNN block dynamically if semantic search is required but missing.
#         """
#         logger.info("No KNN found but semantic search needed, adding KNN query")

#         embedding_vector = await self.generate_embeddings(user_query)
#         if not isinstance(embedding_vector, list):
#             raise ValueError("Embedding vector must be list of floats")

#         knn_query = {
#             "knn": {
#                 "field": "chunk_vector",
#                 "query_vector": embedding_vector,
#                 "k": k,
#                 "num_candidates": num_candidates
#             }
#         }

#         logger.info(f"✅ Added fallback KNN query for: {user_query}")
#         return knn_query

#     async def generate_query(self, user_query: str, source_filter: Optional[str] = None) -> dict:
#         """Generate an OpenSearch query DSL body from a natural language query."""
#         try:
#             schema = self._prepare_schema()
#             prompt = QUERY_GENERATION_PROMPT.format(schema=schema, query=user_query)

#             # Get LLM to generate query structure
#             response_text = self.bedrock.invoke_model(
#                 model_id=self.model_id,
#                 prompt=prompt,
#                 max_tokens=2000,
#                 temperature=0.0
#             )

#             logger.info(f"=== LLM FULL Response ===\n{response_text}\n=== END Response ===")
            
#             query_body = self._extract_json(response_text)
#             logger.info(f"=== Extracted query_body ===\n{json.dumps(query_body, indent=2)}\n=== END query_body ===")

#             # Check if we need semantic search
#             needs_semantic_search = self._needs_semantic_search(user_query)
#             logger.info(f"User query needs semantic search: {needs_semantic_search}")

#             if not query_body or ("query" not in query_body and "knn" not in query_body):
#                 logger.warning(f"Invalid query generated. Response: {response_text[:200]}")
#                 if needs_semantic_search:
#                     # ✅ Add await
#                     logger.info("Generating KNN query directly as fallback")
#                     query_body = await self._create_knn_query(user_query)
#                 else:
#                     query_body = self._default_query()

#             # Process KNN queries - find placeholders and replace with actual embeddings
#             try:
#                 knn_processed = await self._process_knn_embeddings(query_body, user_query)
#                 logger.info(f"KNN embeddings processed: {knn_processed}")
                
#                 # If no KNN was found but query needs semantic search, add it
#                 if not knn_processed and needs_semantic_search:
#                     logger.info("No KNN found but semantic search needed, adding KNN query")
#                     # ✅ Add await
#                     query_body = await self._add_knn_to_query(user_query)
                    
#             except Exception as e:
#                 logger.error(f"Failed to process KNN embeddings: {e}")
#                 if needs_semantic_search:
#                     logger.info("Falling back to direct KNN query creation")
#                     # ✅ Add await
#                     query_body = await self._create_knn_query(user_query)
#                 else:
#                     return self._default_query()

#             # Initialize query structure if it doesn't exist
#             if "query" not in query_body:
#                 if "knn" in query_body:
#                     # KNN at root level is valid for OpenSearch, keep it
#                     pass
#                 else:
#                     query_body["query"] = {"match_all": {}}

#             # Add source filter if provided
#             if source_filter:
#                 query_body = self._apply_source_filter(query_body, source_filter)

#             logger.info(f"Successfully generated OpenSearch query for: {user_query}")
#             if source_filter:
#                 logger.info(f"Source filter applied: {source_filter}")
#             logger.info(f"=== FINAL Query body ===\n{json.dumps(query_body, indent=2)}\n=== END FINAL ===")
#             return query_body

#         except Exception as e:
#             logger.error(f"Error generating OpenSearch query: {e}", exc_info=True)
#             return self._default_query()

#     def _extract_json(self, text: str) -> dict:
#         """Extract JSON from the LLM's response text with robust handling."""
#         text = text.strip()
#         logger.debug(f"Raw LLM output (first 500 chars): {text[:500]}")

#         # Common cleanup for markdown/code fences
#         text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
#         text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
#         text = text.strip("` \n\t")

#         # Try to find JSON object anywhere in text
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if not match:
#             logger.error("No JSON braces found in response.")
#             return {}

#         candidate = match.group(0)
#         try:
#             parsed = json.loads(candidate)
#             return parsed
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to decode JSON: {e}. Candidate snippet: {candidate[:300]}")
#             return {}

#     async def _process_knn_embeddings(self, query_body: dict, user_query: str) -> bool:
#         """
#         Recursively find KNN queries with placeholder text and replace with actual embeddings.
#         Handles KNN queries at any nesting level (root, bool.must, bool.should, etc.)
        
#         Returns:
#             bool: True if at least one KNN query was found and processed
#         """
#         knn_found = False
        
#         async def process_dict(obj, parent_key=None):
#             nonlocal knn_found
#             if isinstance(obj, dict):
#                 # Check if this dict contains a KNN query with placeholder
#                 if "knn" in obj:
#                     knn_query = obj["knn"]
#                     if isinstance(knn_query, dict):
#                         await self._replace_placeholder_with_embedding(knn_query, user_query)
#                         knn_found = True
                
#                 # Recursively process all nested structures
#                 for key, value in obj.items():
#                     if isinstance(value, (dict, list)):
#                         await process_dict(value, key)
            
#             elif isinstance(obj, list):
#                 for item in obj:
#                     if isinstance(item, (dict, list)):
#                         await process_dict(item, parent_key)
        
#         await process_dict(query_body)
#         return knn_found

#     async def _replace_placeholder_with_embedding(self, knn_query: dict, user_query: str) -> None:
#         """
#         Replace placeholder text in KNN query with actual embedding vector.
#         Supports both query_vector_builder format and placeholder format.
#         """
#         try:
#             embedding_text = None
            
#             # Check for placeholder format
#             if "query_vector" in knn_query:
#                 query_vector = knn_query["query_vector"]
#                 if isinstance(query_vector, str) and "__EMBEDDING_TEXT__" in query_vector:
#                     embedding_text = user_query
#                     logger.info(f"Found placeholder, using user query: {user_query}")
#                 elif isinstance(query_vector, str):
#                     # It's a string but not placeholder - might be the actual text to embed
#                     embedding_text = query_vector
#                     logger.info(f"Found text in query_vector: {embedding_text}")
#                 else:
#                     # Already has vector, skip
#                     return
            
#             # Check for query_vector_builder format (legacy support)
#             elif "query_vector_builder" in knn_query:
#                 model_text = knn_query["query_vector_builder"]["text_embedding"]["model_text"]
#                 embedding_text = model_text
#                 logger.info(f"Found query_vector_builder with text: {embedding_text}")
            
#             if not embedding_text:
#                 logger.warning("No embedding text found in KNN query")
#                 return
            
#             # Generate the actual embedding vector
#             embedding_vector = await self.generate_embeddings(embedding_text)
            
#             # Replace with actual vector
#             knn_query.pop("query_vector_builder", None)
#             knn_query["query_vector"] = embedding_vector
            
#             logger.info(f"Successfully replaced placeholder with {len(embedding_vector)}-dim embedding vector")
            
#         except Exception as e:
#             logger.error(f"Failed to replace placeholder with embedding: {e}", exc_info=True)
#             raise

#     def _apply_source_filter(self, query_body: dict, source_filter: str) -> dict:
#         """Apply source filter to the query body."""
#         try:
#             # Handle root-level KNN query
#             if "knn" in query_body and "query" not in query_body:
#                 knn_query = query_body.pop("knn")
#                 if source_filter == "other":
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [{"knn": knn_query}],
#                             "must_not": [{"term": {"source": "court_listener"}}]
#                         }
#                     }
#                 else:
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [
#                                 {"knn": knn_query},
#                                 {"term": {"source": source_filter}}
#                             ]
#                         }
#                     }
#                 return query_body
            
#             # Handle query structure
#             original_query = query_body.get("query", {"match_all": {}})
            
#             # Check if original query is already a bool query
#             if isinstance(original_query, dict) and "bool" in original_query:
#                 bool_query = original_query["bool"]
                
#                 if source_filter == "other":
#                     if "must_not" not in bool_query:
#                         bool_query["must_not"] = []
#                     bool_query["must_not"].append({"term": {"source": "court_listener"}})
#                 else:
#                     if "must" not in bool_query:
#                         bool_query["must"] = []
#                     bool_query["must"].append({"term": {"source": source_filter}})
#             else:
#                 # Wrap original query
#                 if source_filter == "other":
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [original_query],
#                             "must_not": [{"term": {"source": "court_listener"}}]
#                         }
#                     }
#                 else:
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [
#                                 original_query,
#                                 {"term": {"source": source_filter}}
#                             ]
#                         }
#                     }
            
#             return query_body
            
#         except Exception as e:
#             logger.error(f"Error applying source filter: {e}", exc_info=True)
#             return query_body

#     def _default_query(self) -> dict:
#         """Return a fallback match_all query."""
#         return {"query": {"match_all": {}}}








# class OpenSearchQueryGenerator:
#     def __init__(self):
#         self.bedrock = BedrockClient(BedrockConfig())
#         self.model_id = BEDROCK_MODEL
#         self.embedding_model_id = "amazon.titan-embed-text-v2:0"

#     def _prepare_schema(self) -> str:
#         """Prepare schema for the prompt."""
#         return OPENSEARCH_SCHEMA.format(
#             concerns=", ".join(concerns_events) + "...",
#             emerging_risks=", ".join(emerging_risks) + "...",
#             misc_topics=", ".join(misc_topics) + "...",
#             naics_data=", ".join([f"{item['code']}" for item in naics_data]) + "..."
#         )
    
#     async def generate_embeddings(self, text: str) -> List[float]:
#       """
#       Generate embedding vector for given text using Amazon Titan Embedding model.
      
#       Args:
#           text: The text to generate embeddings for
          
#       Returns:
#           List of floats representing the embedding vector (1024 dimensions for Titan v2)
#       """
#       try:
#           logger.info(f"Generating embedding for text: {text[:100]}...")
          
#           async with CommonSession(
#               CommonSessionConfig(
#                   client_name="bedrock-runtime",
#                   region="us-east-1",
#                   profile_name="Comm-Prop-Sandbox"
#               )
#           ) as titan:
#               titan_embedding = TitanV2(titan)
              
#               # ✅ CRITICAL: Pass dict, not JSON string
#               # invoke_with_retry will call json.dumps() internally (base.py line 70)
#               payload = {"inputText": text}
#               response = await titan_embedding.generate_embedding(payload)
              
#               # TitanV2.generate_embedding returns the embedding vector directly
#               # as a list of floats (see base.py line 156 and line 72)
#               if not isinstance(response, list):
#                   logger.error(f"Unexpected response type: {type(response)}, value: {response}")
#                   raise ValueError(f"Expected list of floats, got {type(response)}")
              
#               if len(response) == 0:
#                   raise ValueError("Received empty embedding vector")
              
#               logger.info(f"✅ Successfully generated {len(response)}-dimensional embedding")
#               return response
              
#       except Exception as e:
#           logger.error(f"Failed to generate embedding: {e}", exc_info=True)
#           raise

#     # async def generate_embeddings(self, text: str) :
#     #     """
#     #     Generate embedding vector for given text using Amazon Titan Embedding model.
        
#     #     Args:
#     #         text: The text to generate embeddings for
            
#     #     Returns:
#     #         List of floats representing the embedding vector (1024 dimensions for Titan v2)
#     #     """

#     #     # from CommonService.async_commonsession.commonsession import (
#     #     #     CommonSession,
#     #     #     CommonSessionConfig,
#     #     # )
        
#     #     # async def generate_embed():
#     #     # async with CommonSession(
#     #     #         CommonSessionConfig(
#     #     #             client_name="bedrock-runtime", region="us-east-1", profile_name="Comm-Prop-Sandbox"
#     #     #         )
#     #     #     ) as titan:
#     #     #     titan_embedding = TitanV1(titan)
#     #     #     response = await titan_embedding.generate_embedding({"inputText": "Hello world!"})
#     #     #     print(response)

#     #     try:
#     #         logger.info(f"Generating embedding for text: {text}...")
#     #         async with CommonSession(
#     #             CommonSessionConfig(
#     #                 client_name="bedrock-runtime", region="us-east-1", profile_name="Comm-Prop-Sandbox"
#     #             )
#     #         ) as titan:
#     #           titan_embedding = TitanV2(titan)
#     #           payload = json.dumps({"inputText": text}) 
#     #           # response = await titan_embedding.generate_embedding({"inputText": text})
#     #           response = await titan_embedding.generate_embedding(payload)
#     #           print("type of embeddings", type(response))
#     #           # print(response)
              
#     #         # Call Bedrock to generate embedding
#     #         # response = self.bedrock.invoke_model(
#     #         #     model_id=self.embedding_model_id,
#     #         #     input_text=text
#     #         # )

#     #         # response = self.bedrock.invoke_model(
#     #         #     model_id=self.embedding_model_id,
#     #         #     body={"inputText": text}   # ✅ Correct parameter for Titan embeddings
#     #         # )
            
#     #         # Extract embedding vector from response
#     #         if isinstance(response, dict):
#     #             if "embedding" in response:
#     #                 embedding_vector = response["embedding"]
#     #             elif "embeddings" in response:
#     #                 embedding_vector = response["embeddings"][0] if isinstance(response["embeddings"], list) else response["embeddings"]
#     #             else:
#     #                 logger.error(f"Unexpected embedding response format: {response}")
#     #                 raise ValueError("Could not extract embedding from response")
#     #         elif isinstance(response, list):
#     #             embedding_vector = response
#     #         else:
#     #             logger.error(f"Unexpected response type: {type(response)}")
#     #             raise ValueError("Invalid embedding response type")
            
#     #         # Validate embedding
#     #         if not isinstance(embedding_vector, list) or len(embedding_vector) == 0:
#     #             raise ValueError(f"Invalid embedding vector: {type(embedding_vector)}, length: {len(embedding_vector) if isinstance(embedding_vector, list) else 'N/A'}")
            
#     #         logger.info(f"Successfully generated {len(embedding_vector)}-dimensional embedding")
#     #         return embedding_vector
            
#     #     except Exception as e:
#     #         logger.error(f"Failed to generate embedding: {e}", exc_info=True)
#     #         raise
        
    
#     async def _create_knn_query(self, user_query: str):
#       """
#       Build a proper KNN query for OpenSearch using the 1024-dim embedding.
#       """
#       embedding_vector = await self.generate_embeddings(user_query)
#       print("--------------------------------",type(embedding_vector))
#       return {
#           "knn": {
#               "field": "chunk_vector",
#               "query_vector": embedding_vector,
#               "k": 10,
#               "num_candidates": 100
#           }
#       }

    
#     def _needs_semantic_search(self, user_query: str) -> bool:
#           """
#           Determine whether the user query should trigger a semantic (vector) search.
#           Returns True if the query likely targets 'data', 'reason_identified', or 'title'
#           fields, or contains general article-topic language.
#           """
#           if not user_query:
#               return False

#           query_lower = user_query.lower()
#           semantic_keywords = [
#               "data", "reason_identified", "title",
#               "about", "related to", "discuss", "concern",
#               "impact", "risk", "article", "topic"
#           ]

#           # If any keyword appears, we assume semantic search is needed
#           return any(keyword in query_lower for keyword in semantic_keywords)


#     # async def _add_knn_to_query(self, user_query: str, k: int = 10, num_candidates: int = 100):
#     #   """
#     #   Add KNN block dynamically if semantic search is required but missing.
#     #   """
#     #   logger.info("No KNN found but semantic search needed, adding KNN query")

#     #   embedding_vector = await self.generate_embeddings(user_query)
#     #   if not isinstance(embedding_vector, list):
#     #       raise ValueError("Embedding vector must be list of floats")

#     #   knn_query = {
#     #       "knn": {
#     #           "field": "chunk_vector",
#     #           "query_vector": embedding_vector,
#     #           "k": k,
#     #           "num_candidates": num_candidates
#     #       }
#     #   }

#     #   logger.info(f"✅ Added fallback KNN query for: {user_query}")
#     #   return knn_query


#     # async def generate_query(self, user_query: str, source_filter: Optional[str] = None) -> dict:
#     #     """Generate an OpenSearch query DSL body from a natural language query."""
#     #     try:
#     #         schema = self._prepare_schema()
#     #         prompt = QUERY_GENERATION_PROMPT.format(schema=schema, query=user_query)

#     #         # Get LLM to generate query structure
#     #         response_text = self.bedrock.invoke_model(
#     #             model_id=self.model_id,
#     #             prompt=prompt,
#     #             max_tokens=2000,
#     #             temperature=0.0
#     #         )

#     #         logger.info(f"=== LLM FULL Response ===\n{response_text}\n=== END Response ===")
            
#     #         query_body = self._extract_json(response_text)
#     #         logger.info(f"=== Extracted query_body ===\n{json.dumps(query_body, indent=2)}\n=== END query_body ===")

#     #         # Check if we need semantic search
#     #         needs_semantic_search = self._needs_semantic_search(user_query)
#     #         logger.info(f"User query needs semantic search: {needs_semantic_search}")

#     #         if not query_body or ("query" not in query_body and "knn" not in query_body):
#     #             logger.warning(f"Invalid query generated. Response: {response_text[:200]}")
#     #             if needs_semantic_search:
#     #                 # Generate KNN query directly
#     #                 logger.info("Generating KNN query directly as fallback")
#     #                 query_body = await self._create_knn_query(user_query)
#     #             else:
#     #                 query_body = self._default_query()

#     #         # Process KNN queries - find placeholders and replace with actual embeddings
#     #         try:
#     #             knn_processed = await self._process_knn_embeddings(query_body, user_query)
#     #             logger.info(f"KNN embeddings processed: {knn_processed}")
                
#     #             # If no KNN was found but query needs semantic search, add it
#     #             if not knn_processed and needs_semantic_search:
#     #                 logger.info("No KNN found but semantic search needed, adding KNN query")
#     #                 query_body = await self._add_knn_to_query(query_body, user_query)
                    
#     #         except Exception as e:
#     #             logger.error(f"Failed to process KNN embeddings: {e}")
#     #             if needs_semantic_search:
#     #                 logger.info("Falling back to direct KNN query creation")
#     #                 query_body = await self._create_knn_query(user_query)
#     #             else:
#     #                 return self._default_query()

#     #         # Initialize query structure if it doesn't exist
#     #         if "query" not in query_body:
#     #             if "knn" in query_body:
#     #                 # KNN at root level is valid for OpenSearch, keep it
#     #                 pass
#     #             else:
#     #                 query_body["query"] = {"match_all": {}}

#     #         # Add source filter if provided
#     #         if source_filter:
#     #             query_body = self._apply_source_filter(query_body, source_filter)

#     #         logger.info(f"Successfully generated OpenSearch query for: {user_query}")
#     #         if source_filter:
#     #             logger.info(f"Source filter applied: {source_filter}")
#     #         logger.info(f"=== FINAL Query body ===\n{json.dumps(query_body, indent=2)}\n=== END FINAL ===")
#     #         return query_body

#     #     except Exception as e:
#     #         logger.error(f"Error generating OpenSearch query: {e}", exc_info=True)
#     #         return self._default_query()

    
#     async def generate_query(self, user_query: str, source_filter: Optional[str] = None) -> dict:
#       """Generate an OpenSearch query DSL body from a natural language query."""
#       try:
#           schema = self._prepare_schema()
#           prompt = QUERY_GENERATION_PROMPT.format(schema=schema, query=user_query)

#           # Get LLM to generate query structure
#           response_text = self.bedrock.invoke_model(
#               model_id=self.model_id,
#               prompt=prompt,
#               max_tokens=2000,
#               temperature=0.0
#           )

#           logger.info(f"=== LLM FULL Response ===\n{response_text}\n=== END Response ===")
          
#           query_body = self._extract_json(response_text)
#           logger.info(f"=== Extracted query_body ===\n{json.dumps(query_body, indent=2)}\n=== END query_body ===")

#           # Check if we need semantic search
#           needs_semantic_search = self._needs_semantic_search(user_query)
#           logger.info(f"User query needs semantic search: {needs_semantic_search}")

#           if not query_body or ("query" not in query_body and "knn" not in query_body):
#               logger.warning(f"Invalid query generated. Response: {response_text[:200]}")
#               if needs_semantic_search:
#                   # ✅ FIX 1: Add await here
#                   logger.info("Generating KNN query directly as fallback")
#                   query_body = await self._create_knn_query(user_query)
#               else:
#                   query_body = self._default_query()

#           # Process KNN queries - find placeholders and replace with actual embeddings
#           try:
#               knn_processed = await self._process_knn_embeddings(query_body, user_query)
#               logger.info(f"KNN embeddings processed: {knn_processed}")
              
#               # If no KNN was found but query needs semantic search, add it
#               if not knn_processed and needs_semantic_search:
#                   logger.info("No KNN found but semantic search needed, adding KNN query")
#                   # ✅ FIX 2: await the call, and receive query_body return value
#                   query_body = await self._add_knn_to_query(user_query)
                  
#           except Exception as e:
#               logger.error(f"Failed to process KNN embeddings: {e}")
#               if needs_semantic_search:
#                   logger.info("Falling back to direct KNN query creation")
#                   # ✅ FIX 3: Add await here too
#                   query_body = await self._create_knn_query(user_query)
#               else:
#                   return self._default_query()

#           # Initialize query structure if it doesn't exist
#           if "query" not in query_body:
#               if "knn" in query_body:
#                   # KNN at root level is valid for OpenSearch, keep it
#                   pass
#               else:
#                   query_body["query"] = {"match_all": {}}

#           # Add source filter if provided
#           if source_filter:
#               query_body = self._apply_source_filter(query_body, source_filter)

#           logger.info(f"Successfully generated OpenSearch query for: {user_query}")
#           if source_filter:
#               logger.info(f"Source filter applied: {source_filter}")
#           logger.info(f"=== FINAL Query body ===\n{json.dumps(query_body, indent=2)}\n=== END FINAL ===")
#           return query_body

#       except Exception as e:
#           logger.error(f"Error generating OpenSearch query: {e}", exc_info=True)
#           return self._default_query()


#     async def _add_knn_to_query(self, user_query: str, k: int = 10, num_candidates: int = 100):
#       """
#       Add KNN block dynamically if semantic search is required but missing.
#       """
#       logger.info("No KNN found but semantic search needed, adding KNN query")

#       embedding_vector = await self.generate_embeddings(user_query)
#       if not isinstance(embedding_vector, list):
#           raise ValueError("Embedding vector must be list of floats")

#       knn_query = {
#           "knn": {
#               "field": "chunk_vector",
#               "query_vector": embedding_vector,
#               "k": k,
#               "num_candidates": num_candidates
#           }
#       }

#       logger.info(f"✅ Added fallback KNN query for: {user_query}")
#       return knn_query


#     async def _process_knn_embeddings(self, query_body: dict, user_query: str) -> bool:
#       """
#       Recursively find KNN queries with placeholder text and replace with actual embeddings.
#       Handles KNN queries at any nesting level (root, bool.must, bool.should, etc.)
      
#       Returns:
#           bool: True if at least one KNN query was found and processed
#       """
#       knn_found = False
      
#       async def process_dict(obj, parent_key=None):
#           nonlocal knn_found
#           if isinstance(obj, dict):
#               # Check if this dict contains a KNN query with placeholder
#               if "knn" in obj:
#                   knn_query = obj["knn"]
#                   if isinstance(knn_query, dict):
#                       await self._replace_placeholder_with_embedding(knn_query, user_query)
#                       knn_found = True
              
#               # Recursively process all nested structures
#               for key, value in obj.items():
#                   if isinstance(value, (dict, list)):
#                       await process_dict(value, key)
          
#           elif isinstance(obj, list):
#               for item in obj:
#                   if isinstance(item, (dict, list)):
#                       await process_dict(item, parent_key)
      
#       await process_dict(query_body)
#       return knn_found



#     def _extract_json(self, text: str) -> dict:
#         """Extract JSON from the LLM's response text with robust handling."""
#         text = text.strip()
#         logger.debug(f"Raw LLM output (first 500 chars): {text[:500]}")

#         # Common cleanup for markdown/code fences
#         text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
#         text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
#         text = text.strip("` \n\t")

#         # Try to find JSON object anywhere in text
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if not match:
#             logger.error("No JSON braces found in response.")
#             return {}

#         candidate = match.group(0)
#         try:
#             parsed = json.loads(candidate)
#             return parsed
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to decode JSON: {e}. Candidate snippet: {candidate[:300]}")
#             return {}

#     # async def _process_knn_embeddings(self, query_body: dict, user_query: str) -> None:
#     #     """
#     #     Recursively find KNN queries with placeholder text and replace with actual embeddings.
#     #     Handles KNN queries at any nesting level (root, bool.must, bool.should, etc.)
#     #     """
#     #     async def process_dict(obj, parent_key=None):
#     #         if isinstance(obj, dict):
#     #             # Check if this dict contains a KNN query with placeholder
#     #             if "knn" in obj:
#     #                 knn_query = obj["knn"]
#     #                 if isinstance(knn_query, dict):
#     #                     await self._replace_placeholder_with_embedding(knn_query, user_query)
                
#     #             # Recursively process all nested structures
#     #             for key, value in obj.items():
#     #                 if isinstance(value, (dict, list)):
#     #                     await process_dict(value, key)
            
#     #         elif isinstance(obj, list):
#     #             for item in obj:
#     #                 if isinstance(item, (dict, list)):
#     #                     await process_dict(item, parent_key)
        
#     #     await process_dict(query_body)

#     async def _replace_placeholder_with_embedding(self, knn_query: dict, user_query: str) -> None:
#         """
#         Replace placeholder text in KNN query with actual embedding vector.
#         Supports both query_vector_builder format and placeholder format.
#         """
#         try:
#             embedding_text = None
            
#             # Check for placeholder format
#             if "query_vector" in knn_query:
#                 query_vector = knn_query["query_vector"]
#                 if isinstance(query_vector, str) and "__EMBEDDING_TEXT__" in query_vector:
#                     embedding_text = user_query
#                     logger.info(f"Found placeholder, using user query: {user_query}")
#                 elif isinstance(query_vector, str):
#                     # It's a string but not placeholder - might be the actual text to embed
#                     embedding_text = query_vector
#                     logger.info(f"Found text in query_vector: {embedding_text}")
#                 else:
#                     # Already has vector, skip
#                     return
            
#             # Check for query_vector_builder format (legacy support)
#             elif "query_vector_builder" in knn_query:
#                 model_text = knn_query["query_vector_builder"]["text_embedding"]["model_text"]
#                 embedding_text = model_text
#                 logger.info(f"Found query_vector_builder with text: {embedding_text}")
            
#             if not embedding_text:
#                 logger.warning("No embedding text found in KNN query")
#                 return
            
#             # Generate the actual embedding vector
#             embedding_vector = await self.generate_embeddings(embedding_text)
            
#             # Replace with actual vector
#             knn_query.pop("query_vector_builder", None)
#             knn_query["query_vector"] = embedding_vector
            
#             logger.info(f"Successfully replaced placeholder with {len(embedding_vector)}-dim embedding vector")
            
#         except Exception as e:
#             logger.error(f"Failed to replace placeholder with embedding: {e}", exc_info=True)
#             raise

#     def _apply_source_filter(self, query_body: dict, source_filter: str) -> dict:
#         """Apply source filter to the query body."""
#         try:
#             # Handle root-level KNN query
#             if "knn" in query_body and "query" not in query_body:
#                 knn_query = query_body.pop("knn")
#                 if source_filter == "other":
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [{"knn": knn_query}],
#                             "must_not": [{"term": {"source": "court_listener"}}]
#                         }
#                     }
#                 else:
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [
#                                 {"knn": knn_query},
#                                 {"term": {"source": source_filter}}
#                             ]
#                         }
#                     }
#                 return query_body
            
#             # Handle query structure
#             original_query = query_body.get("query", {"match_all": {}})
            
#             # Check if original query is already a bool query
#             if isinstance(original_query, dict) and "bool" in original_query:
#                 bool_query = original_query["bool"]
                
#                 if source_filter == "other":
#                     if "must_not" not in bool_query:
#                         bool_query["must_not"] = []
#                     bool_query["must_not"].append({"term": {"source": "court_listener"}})
#                 else:
#                     if "must" not in bool_query:
#                         bool_query["must"] = []
#                     bool_query["must"].append({"term": {"source": source_filter}})
#             else:
#                 # Wrap original query
#                 if source_filter == "other":
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [original_query],
#                             "must_not": [{"term": {"source": "court_listener"}}]
#                         }
#                     }
#                 else:
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [
#                                 original_query,
#                                 {"term": {"source": source_filter}}
#                             ]
#                         }
#                     }
            
#             return query_body
            
#         except Exception as e:
#             logger.error(f"Error applying source filter: {e}", exc_info=True)
#             return query_body

#     def _default_query(self) -> dict:
#         """Return a fallback match_all query."""
#         return {"query": {"match_all": {}}}










# import json
# import re
# from typing import Optional, List
# from src.api.routes.bedrock_client import BedrockClient, BedrockConfig
# from src.api.routes.logger import get_logger
# from src.api.routes.concern_risk_misc_naics import concerns_events, emerging_risks, misc_topics, naics_data
# from src.api.routes.settings import BEDROCK_MODEL

# logger = get_logger(__name__)

# # ------------------ OPENSEARCH SCHEMA ------------------

# OPENSEARCH_SCHEMA = """ 
# Index Name: ei_articles_index-05-nov-test

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

# # ------------------ PROMPT TEMPLATE ------------------

# QUERY_GENERATION_PROMPT = """
# You are an expert at converting natural language queries into valid OpenSearch DSL queries.

# INDEX SCHEMA:
# {schema}

# USER QUERY: {query}

# CRITICAL INSTRUCTIONS:
# 1. Generate ONLY a valid JSON object for the OpenSearch query body.
# 2. Use EXACT field names from the schema (case-sensitive).
# 3. Query rules:
#    - If the query involves *content*, *title*, *reason*, or any semantic topic (e.g. mentions data, reason_identified, title, or general article topic):
#        → Use a similarity (vector) search on `chunk_vector` with a KNN query.
#        → Use the placeholder text "__EMBEDDING_TEXT__" for the semantic search text
#        Example:
#        {{
#          "knn": {{
#            "field": "chunk_vector",
#            "query_vector": "__EMBEDDING_TEXT__",
#            "k": 10,
#            "num_candidates": 100
#          }}
#        }}
#    - For metadata searches (e.g. tag, source, region, concerns, emerging_risk_name, miscTopics, naicscode):
#        → Use `term`, `terms`, `match`, or `range` queries on those fields.
#    - To combine similarity and filters (hybrid search):
#        → Use a `bool` query where `must` includes both the `knn` and any filters.
#        ✅ The `knn` query should be placed directly in `must` array.

# 4. Field types:
#    - Keyword: tag, source, concerns, emerging_risk_name, miscTopics, naicscode, naics_description, region, url
#    - Text: title, description, data, reason_identified, chunk_text
#    - Boolean: is_latest
#    - Numeric: doc_id, chunk_id
#    - Date/time: published_time, last_update_time, injection_time

# 5. Common query patterns:
#    - "Show all articles" → match_all
#    - "Tagged as Current" → term query on tag
#    - "From Reuters" → term query on source
#    - "With concern PFAS" → term query on concerns
#    - "Show recent articles" → range query on published_time or last_update_time

# 6. Combination examples:
#    - "Articles about climate change tagged as Current":
#      → Use `bool.must` with knn query for "climate change" and a term query for tag="Current".
#    - "Emerging risks in Europe":
#      → term on region="Europe" + knn if context indicates semantic topic.
#    - "Show PFAS untagged articles":
#      → bool.must with term(tag="Untagged") and term(concerns="PFAS").
#    - "Articles about wildfires in last 3 days":
#      → bool.must with range on published_time and knn for "wildfires".

# 7. Always structure hybrid queries like this:
#    {{
#      "query": {{
#        "bool": {{
#          "must": [
#            <other field filters>,
#            {{
#              "knn": {{
#                "field": "chunk_vector",
#                "query_vector": "__EMBEDDING_TEXT__",
#                "k": 10,
#                "num_candidates": 100
#              }}
#            }}
#          ]
#        }}
#      }}
#    }}

# 8. Return ONLY the JSON object — no text, markdown, or explanations.

# ---

# EXAMPLES:

# Query: "Show me all articles tagged as Current"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Current"
#     }}
#   }}
# }}

# Query: "Find articles about climate change"
# {{
#   "knn": {{
#     "field": "chunk_vector",
#     "query_vector": "__EMBEDDING_TEXT__",
#     "k": 10,
#     "num_candidates": 100
#   }}
# }}

# Query: "Find articles about lawsuits tagged as Current"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "term": {{
#             "tag": "Current"
#           }}
#         }},
#         {{
#           "knn": {{
#             "field": "chunk_vector",
#             "query_vector": "__EMBEDDING_TEXT__",
#             "k": 10,
#             "num_candidates": 100
#           }}
#         }}
#       ]
#     }}
#   }}
# }}

# Query: "Show untagged articles"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Untagged"
#     }}
#   }}
# }}

# Query: "Show articles with wildfire for last 3 days"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "range": {{
#             "published_time": {{
#               "gte": "now-3d/d",
#               "lte": "now"
#             }}
#           }}
#         }},
#         {{
#           "knn": {{
#             "field": "chunk_vector",
#             "query_vector": "__EMBEDDING_TEXT__",
#             "k": 10,
#             "num_candidates": 100
#           }}
#         }}
#       ]
#     }}
#   }}
# }}
# """


# class OpenSearchQueryGenerator:
#     def __init__(self):
#         self.bedrock = BedrockClient(BedrockConfig())
#         self.model_id = BEDROCK_MODEL
#         self.embedding_model_id = "amazon.titan-embed-text-v2:0"

#     def _prepare_schema(self) -> str:
#         """Prepare schema for the prompt."""
#         return OPENSEARCH_SCHEMA.format(
#             concerns=", ".join(concerns_events) + "...",
#             emerging_risks=", ".join(emerging_risks) + "...",
#             misc_topics=", ".join(misc_topics) + "...",
#             naics_data=", ".join([f"{item['code']}" for item in naics_data]) + "..."
#         )

#     def generate_embedding(self, text: str) -> List[float]:
#         """
#         Generate embedding vector for given text using Amazon Titan Embedding model.
        
#         Args:
#             text: The text to generate embeddings for
            
#         Returns:
#             List of floats representing the embedding vector (1024 dimensions for Titan v2)
#         """
#         try:
#             logger.info(f"Generating embedding for text: {text[:100]}...")
            
#             # Call Bedrock to generate embedding
#             response = self.bedrock.invoke_model(
#                 model_id=self.embedding_model_id,
#                 input_text=text
#             )
            
#             # Extract embedding vector from response
#             if isinstance(response, dict):
#                 if "embedding" in response:
#                     embedding_vector = response["embedding"]
#                 elif "embeddings" in response:
#                     embedding_vector = response["embeddings"][0] if isinstance(response["embeddings"], list) else response["embeddings"]
#                 else:
#                     logger.error(f"Unexpected embedding response format: {response}")
#                     raise ValueError("Could not extract embedding from response")
#             elif isinstance(response, list):
#                 embedding_vector = response
#             else:
#                 logger.error(f"Unexpected response type: {type(response)}")
#                 raise ValueError("Invalid embedding response type")
            
#             # Validate embedding
#             if not isinstance(embedding_vector, list) or len(embedding_vector) == 0:
#                 raise ValueError(f"Invalid embedding vector: {type(embedding_vector)}, length: {len(embedding_vector) if isinstance(embedding_vector, list) else 'N/A'}")
            
#             logger.info(f"Successfully generated {len(embedding_vector)}-dimensional embedding")
#             return embedding_vector
            
#         except Exception as e:
#             logger.error(f"Failed to generate embedding: {e}", exc_info=True)
#             raise

#     def generate_query(self, user_query: str, source_filter: Optional[str] = None) -> dict:
#         """Generate an OpenSearch query DSL body from a natural language query."""
#         try:
#             schema = self._prepare_schema()
#             prompt = QUERY_GENERATION_PROMPT.format(schema=schema, query=user_query)

#             # Get LLM to generate query structure
#             response_text = self.bedrock.invoke_model(
#                 model_id=self.model_id,
#                 prompt=prompt,
#                 max_tokens=2000,
#                 temperature=0.0
#             )

#             logger.info(f"LLM Response (first 500 chars): {response_text[:500]}...")
            
#             query_body = self._extract_json(response_text)

#             if not query_body or ("query" not in query_body and "knn" not in query_body):
#                 logger.warning(f"Invalid query generated, using match_all. Response: {response_text[:200]}")
#                 query_body = self._default_query()

#             # Process KNN queries - find placeholders and replace with actual embeddings
#             try:
#                 self._process_knn_embeddings(query_body, user_query)
#             except Exception as e:
#                 logger.error(f"Failed to process KNN embeddings: {e}")
#                 return self._default_query()

#             # Initialize query structure if it doesn't exist
#             if "query" not in query_body:
#                 if "knn" in query_body:
#                     # KNN at root level is valid for OpenSearch, keep it
#                     pass
#                 else:
#                     query_body["query"] = {"match_all": {}}

#             # Add source filter if provided
#             if source_filter:
#                 query_body = self._apply_source_filter(query_body, source_filter)

#             logger.info(f"Successfully generated OpenSearch query for: {user_query}")
#             if source_filter:
#                 logger.info(f"Source filter applied: {source_filter}")
#             logger.info(f"Query body: {json.dumps(query_body, indent=2)}")
#             return query_body

#         except Exception as e:
#             logger.error(f"Error generating OpenSearch query: {e}", exc_info=True)
#             return self._default_query()

#     def _extract_json(self, text: str) -> dict:
#         """Extract JSON from the LLM's response text with robust handling."""
#         text = text.strip()
#         logger.debug(f"Raw LLM output (first 500 chars): {text[:500]}")

#         # Common cleanup for markdown/code fences
#         text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
#         text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
#         text = text.strip("` \n\t")

#         # Try to find JSON object anywhere in text
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if not match:
#             logger.error("No JSON braces found in response.")
#             return {}

#         candidate = match.group(0)
#         try:
#             parsed = json.loads(candidate)
#             return parsed
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to decode JSON: {e}. Candidate snippet: {candidate[:300]}")
#             return {}

#     def _process_knn_embeddings(self, query_body: dict, user_query: str) -> None:
#         """
#         Recursively find KNN queries with placeholder text and replace with actual embeddings.
#         Handles KNN queries at any nesting level (root, bool.must, bool.should, etc.)
#         """
#         def process_dict(obj, parent_key=None):
#             if isinstance(obj, dict):
#                 # Check if this dict contains a KNN query with placeholder
#                 if "knn" in obj:
#                     knn_query = obj["knn"]
#                     if isinstance(knn_query, dict):
#                         self._replace_placeholder_with_embedding(knn_query, user_query)
                
#                 # Recursively process all nested structures
#                 for key, value in obj.items():
#                     if isinstance(value, (dict, list)):
#                         process_dict(value, key)
            
#             elif isinstance(obj, list):
#                 for item in obj:
#                     if isinstance(item, (dict, list)):
#                         process_dict(item, parent_key)
        
#         process_dict(query_body)

#     def _replace_placeholder_with_embedding(self, knn_query: dict, user_query: str) -> None:
#         """
#         Replace placeholder text in KNN query with actual embedding vector.
#         Supports both query_vector_builder format and placeholder format.
#         """
#         try:
#             embedding_text = None
            
#             # Check for placeholder format
#             if "query_vector" in knn_query:
#                 query_vector = knn_query["query_vector"]
#                 if isinstance(query_vector, str) and "__EMBEDDING_TEXT__" in query_vector:
#                     embedding_text = user_query
#                     logger.info(f"Found placeholder, using user query: {user_query}")
#                 elif isinstance(query_vector, str):
#                     # It's a string but not placeholder - might be the actual text to embed
#                     embedding_text = query_vector
#                     logger.info(f"Found text in query_vector: {embedding_text}")
#                 else:
#                     # Already has vector, skip
#                     return
            
#             # Check for query_vector_builder format (legacy support)
#             elif "query_vector_builder" in knn_query:
#                 model_text = knn_query["query_vector_builder"]["text_embedding"]["model_text"]
#                 embedding_text = model_text
#                 logger.info(f"Found query_vector_builder with text: {embedding_text}")
            
#             if not embedding_text:
#                 logger.warning("No embedding text found in KNN query")
#                 return
            
#             # Generate the actual embedding vector
#             embedding_vector = self.generate_embedding(embedding_text)
            
#             # Replace with actual vector
#             knn_query.pop("query_vector_builder", None)
#             knn_query["query_vector"] = embedding_vector
            
#             logger.info(f"Successfully replaced placeholder with {len(embedding_vector)}-dim embedding vector")
            
#         except Exception as e:
#             logger.error(f"Failed to replace placeholder with embedding: {e}", exc_info=True)
#             raise

#     def _apply_source_filter(self, query_body: dict, source_filter: str) -> dict:
#         """Apply source filter to the query body."""
#         try:
#             # Handle root-level KNN query
#             if "knn" in query_body and "query" not in query_body:
#                 knn_query = query_body.pop("knn")
#                 if source_filter == "other":
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [{"knn": knn_query}],
#                             "must_not": [{"term": {"source": "court_listener"}}]
#                         }
#                     }
#                 else:
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [
#                                 {"knn": knn_query},
#                                 {"term": {"source": source_filter}}
#                             ]
#                         }
#                     }
#                 return query_body
            
#             # Handle query structure
#             original_query = query_body.get("query", {"match_all": {}})
            
#             # Check if original query is already a bool query
#             if isinstance(original_query, dict) and "bool" in original_query:
#                 bool_query = original_query["bool"]
                
#                 if source_filter == "other":
#                     if "must_not" not in bool_query:
#                         bool_query["must_not"] = []
#                     bool_query["must_not"].append({"term": {"source": "court_listener"}})
#                 else:
#                     if "must" not in bool_query:
#                         bool_query["must"] = []
#                     bool_query["must"].append({"term": {"source": source_filter}})
#             else:
#                 # Wrap original query
#                 if source_filter == "other":
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [original_query],
#                             "must_not": [{"term": {"source": "court_listener"}}]
#                         }
#                     }
#                 else:
#                     query_body["query"] = {
#                         "bool": {
#                             "must": [
#                                 original_query,
#                                 {"term": {"source": source_filter}}
#                             ]
#                         }
#                     }
            
#             return query_body
            
#         except Exception as e:
#             logger.error(f"Error applying source filter: {e}", exc_info=True)
#             return query_body

#     def _default_query(self) -> dict:
#         """Return a fallback match_all query."""
#         return {"query": {"match_all": {}}}


















# import json
# import re
# from typing import Optional
# from src.api.routes.bedrock_client import BedrockClient, BedrockConfig
# from src.api.routes.logger import get_logger
# from src.api.routes.concern_risk_misc_naics import concerns_events, emerging_risks, misc_topics, naics_data
# from src.api.routes.settings import BEDROCK_MODEL

# logger = get_logger(__name__)

# # ------------------ OPENSEARCH SCHEMA ------------------

# OPENSEARCH_SCHEMA = """ 
# Index Name: ei_articles_index-05-nov-test

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

# # ------------------ PROMPT TEMPLATE ------------------

# QUERY_GENERATION_PROMPT = """
# You are an expert at converting natural language queries into valid OpenSearch DSL queries.

# INDEX SCHEMA:
# {schema}

# USER QUERY: {query}

# CRITICAL INSTRUCTIONS:
# 1. Generate ONLY a valid JSON object for the OpenSearch query body.
# 2. Use EXACT field names from the schema (case-sensitive).
# 3. Query rules:
#    - If the query involves *content*, *title*, *reason*, or any semantic topic (e.g. mentions data, reason_identified, title, or general article topic):
#        → Use a similarity (vector) search on `chunk_vector` with a KNN query.
#        Example:
#        {{
#          "knn": {{
#            "field": "chunk_vector",
#            "query_vector_builder": {{
#              "text_embedding": {{
#                "model_id": "embedding_model_id",
#                "model_text": "<USER_QUERY_TEXT>"
#              }}
#            }},
#            "k": 10,
#            "num_candidates": 100
#          }}
#        }}
#    - For metadata searches (e.g. tag, source, region, concerns, emerging_risk_name, miscTopics, naicscode):
#        → Use `term`, `terms`, `match`, or `range` queries on those fields.
#    - To combine similarity and filters (hybrid search):
#        → Use a `bool` query where `must` includes both the `knn` and any filters.
#        ✅ Do NOT place `knn` inside `filter`; it must be directly under `must`.

# 4. Field types:
#    - Keyword: tag, source, concerns, emerging_risk_name, miscTopics, naicscode, naics_description, region, url
#    - Text: title, description, data, reason_identified, chunk_text
#    - Boolean: is_latest
#    - Numeric: doc_id, chunk_id
#    - Date/time: published_time, last_update_time, injection_time

# 5. Common query patterns:
#    - "Show all articles" → match_all
#    - "Tagged as Current" → term query on tag
#    - "From Reuters" → term query on source
#    - "With concern PFAS" → term query on concerns
#    - "Show recent articles" → range query on published_time or last_update_time

# 6. Combination examples:
#    - "Articles about climate change tagged as Current":
#      → Use `bool.must` with knn query for "climate change" and a term query for tag="Current".
#    - "Emerging risks in Europe":
#      → term on region="Europe" + knn if context indicates semantic topic.
#    - "Show PFAS untagged articles":
#      → bool.must with term(tag="Untagged") and term(concerns="PFAS").
#    - "Articles about wildfires in last 3 days":
#      → bool.must with range on published_time and knn for "wildfires".

# 7. Always structure hybrid queries like this:
#    {{
#      "query": {{
#        "bool": {{
#          "must": [
#            <other field filters>,
#            {{
#              "knn": {{
#                "field": "chunk_vector",
#                "query_vector_builder": {{
#                  "text_embedding": {{
#                    "model_id": "embedding_model_id",
#                    "model_text": "<USER_QUERY_TEXT>"
#                  }}
#                }},
#                "k": 10,
#                "num_candidates": 100
#              }}
#            }}
#          ]
#        }}
#      }}
#    }}

# 8. Return ONLY the JSON object — no text, markdown, or explanations.

# ---

# EXAMPLES:

# Query: "Show me all articles tagged as Current"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Current"
#     }}
#   }}
# }}

# Query: "Find articles about climate change"
# {{
#   "knn": {{
#     "field": "chunk_vector",
#     "query_vector_builder": {{
#       "text_embedding": {{
#         "model_id": "embedding_model_id",
#         "model_text": "climate change"
#       }}
#     }},
#     "k": 10,
#     "num_candidates": 100
#   }}
# }}

# Query: "Find articles about lawsuits tagged as Current"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "term": {{
#             "tag": "Current"
#           }}
#         }},
#         {{
#           "knn": {{
#             "field": "chunk_vector",
#             "query_vector_builder": {{
#               "text_embedding": {{
#                 "model_id": "embedding_model_id",
#                 "model_text": "lawsuits"
#               }}
#             }},
#             "k": 10,
#             "num_candidates": 100
#           }}
#         }}
#       ]
#     }}
#   }}
# }}

# Query: "Show untagged articles"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Untagged"
#     }}
#   }}
# }}

# Query: "Show articles with wildfire for last 3 days"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "range": {{
#             "published_time": {{
#               "gte": "now-3d/d",
#               "lte": "now"
#             }}
#           }}
#         }},
#         {{
#           "knn": {{
#             "field": "chunk_vector",
#             "query_vector_builder": {{
#               "text_embedding": {{
#                 "model_id": "embedding_model_id",
#                 "model_text": "wildfire"
#               }}
#             }},
#             "k": 10,
#             "num_candidates": 100
#           }}
#         }}
#       ]
#     }}
#   }}
# }}
# """


# class OpenSearchQueryGenerator:
#     def __init__(self):
#         self.bedrock = BedrockClient(BedrockConfig())
#         self.model_id = BEDROCK_MODEL

#     def _prepare_schema(self) -> str:
#         """Prepare schema for the prompt."""
#         return OPENSEARCH_SCHEMA.format(
#             concerns=", ".join(concerns_events) + "...",
#             emerging_risks=", ".join(emerging_risks) + "...",
#             misc_topics=", ".join(misc_topics) + "...",
#             naics_data=", ".join([f"{item['code']}" for item in naics_data]) + "..."
#         )

#     def generate_query(self, user_query: str, source_filter: Optional[str] = None) -> dict:
#         """Generate an OpenSearch query DSL body from a natural language query."""
#         try:
#             embedding_model_id = "amazon.titan-embed-text-v2:0"
#             schema = self._prepare_schema()
#             prompt = QUERY_GENERATION_PROMPT.format(schema=schema, query=user_query)
#             prompt = prompt.replace("embedding_model_id", embedding_model_id)

#             response_text = self.bedrock.invoke_model(
#                 model_id=self.model_id,
#                 prompt=prompt,
#                 max_tokens=2000,
#                 temperature=0.0
#             )

#             logger.info(f"LLM Response (first 500 chars): {response_text[:500]}...")
            
#             query_body = self._extract_json(response_text)
#             logger.warning(f"LLM output before fallback: {response_text[:400]}")

#             if not query_body or ("query" not in query_body and "knn" not in query_body):
#                 logger.warning(f"Invalid query generated, using match_all. Response: {response_text[:200]}")
#                 query_body = self._default_query()

#             # Handle KNN query embedding generation - check all locations
#             try:
#                 self._process_knn_embeddings(query_body)
#             except Exception as e:
#                 logger.error(f"Failed to process KNN embeddings: {e}")
#                 return self._default_query()

#             # CRITICAL FIX: Ensure query_body has proper structure before applying filters
#             # Initialize query structure if it doesn't exist
#             if "query" not in query_body:
#                 if "knn" in query_body:
#                     # KNN at root level needs to be wrapped
#                     query_body = {"query": {"match_all": {}}}
#                 else:
#                     query_body["query"] = {"match_all": {}}

#             # Add source filter if provided
#             if source_filter:
#                 original_query = query_body.get("query", {"match_all": {}})
                
#                 # Check if original query is already a bool query
#                 if isinstance(original_query, dict) and "bool" in original_query:
#                     # Merge into existing bool query
#                     bool_query = original_query["bool"]
                    
#                     if source_filter == "other":
#                         # Exclude CourtListener - add to must_not
#                         if "must_not" not in bool_query:
#                             bool_query["must_not"] = []
#                         bool_query["must_not"].append({"term": {"source": "court_listener"}})
#                     else:
#                         # Filter for specific source - add to must
#                         if "must" not in bool_query:
#                             bool_query["must"] = []
#                         bool_query["must"].append({"term": {"source": source_filter}})
#                 else:
#                     # Original query is not a bool query - wrap it to preserve the user's query
#                     if source_filter == "other":
#                         # Exclude CourtListener - wrap user query with must_not
#                         query_body["query"] = {
#                             "bool": {
#                                 "must": [original_query],
#                                 "must_not": [{"term": {"source": "court_listener"}}]
#                             }
#                         }
#                     else:
#                         # Filter for specific source - wrap user query with source filter
#                         query_body["query"] = {
#                             "bool": {
#                                 "must": [
#                                     original_query,
#                                     {"term": {"source": source_filter}}
#                                 ]
#                             }
#                         }

#             logger.info(f"Successfully generated OpenSearch query for: {user_query}")
#             if source_filter:
#                 logger.info(f"Source filter applied: {source_filter}")
#             logger.info(f"Query body: {json.dumps(query_body, indent=2)}")
#             return query_body

#         except Exception as e:
#             logger.error(f"Error generating OpenSearch query: {e}", exc_info=True)
#             return self._default_query()

#     def _extract_json(self, text: str) -> dict:
#       """Extract JSON from the LLM's response text with robust handling."""
#       text = text.strip()
#       logger.debug(f"Raw LLM output (first 500 chars): {text[:500]}")

#       # Common cleanup for markdown/code fences
#       text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
#       text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
#       text = text.strip("` \n\t")

#       # Try to find JSON object anywhere in text
#       match = re.search(r"\{.*\}", text, re.DOTALL)
#       if not match:
#           logger.error("No JSON braces found in response.")
#           return {}

#       candidate = match.group(0)
#       try:
#           parsed = json.loads(candidate)
#           return parsed
#       except json.JSONDecodeError as e:
#           logger.error(f"Failed to decode JSON: {e}. Candidate snippet: {candidate[:300]}")
#           return {}


#     # def _extract_json(self, text: str) -> dict:
#     #     """Extract JSON from the LLM's response text with robust error handling."""
#     #     # Remove common markdown artifacts
#     #     text = text.strip()
        
#     #     # Remove markdown code blocks if present
#     #     if '```' in text:
#     #         # Extract content between code blocks
#     #         match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
#     #         if match:
#     #             text = match.group(1).strip()
        
#     #     try:
#     #         # Try direct parsing
#     #         return json.loads(text)
#     #     except json.JSONDecodeError as e:
#     #         logger.warning(f"Direct JSON parse failed: {e}")
            
#     #         try:
#     #             # Find the first complete JSON object
#     #             start = text.find('{')
#     #             if start == -1:
#     #                 logger.error(f"No opening brace found in: {text[:100]}")
#     #                 return {}
                
#     #             # Count braces to find matching closing brace
#     #             brace_count = 0
#     #             in_string = False
#     #             escape_next = False
#     #             end = start
                
#     #             for i in range(start, len(text)):
#     #                 char = text[i]
                    
#     #                 if escape_next:
#     #                     escape_next = False
#     #                     continue
                    
#     #                 if char == '\\':
#     #                     escape_next = True
#     #                     continue
                    
#     #                 if char == '"':
#     #                     in_string = not in_string
#     #                     continue
                    
#     #                 if not in_string:
#     #                     if char == '{':
#     #                         brace_count += 1
#     #                     elif char == '}':
#     #                         brace_count -= 1
#     #                         if brace_count == 0:
#     #                             end = i + 1
#     #                             break
                
#     #             if end > start:
#     #                 json_str = text[start:end]
#     #                 parsed = json.loads(json_str)
#     #                 logger.info("Successfully extracted JSON from text")
#     #                 return parsed
#     #             else:
#     #                 logger.error("Could not find matching closing brace")
                    
#     #         except json.JSONDecodeError as e2:
#     #             logger.error(f"Failed to extract valid JSON: {e2}")
#     #             logger.error(f"Attempted to parse: {text[start:min(start+200, len(text))]}...")
#     #         except Exception as e3:
#     #             logger.error(f"Unexpected error during JSON extraction: {e3}")
        
#     #     logger.error("All JSON extraction attempts failed, returning empty dict")
#     #     return {}

#     def _process_knn_embeddings(self, query_body: dict) -> None:
#         """
#         Recursively find and replace query_vector_builder with actual embeddings.
#         Handles KNN queries at any nesting level (root, bool.must, bool.should, etc.)
#         """
#         def process_dict(obj):
#             if isinstance(obj, dict):
#                 # Check if this dict contains a KNN query with query_vector_builder
#                 if "knn" in obj:
#                     knn_query = obj["knn"]
#                     if isinstance(knn_query, dict) and "query_vector_builder" in knn_query:
#                         self._replace_with_embedding(knn_query)
                
#                 # Recursively process all nested structures
#                 for key, value in obj.items():
#                     if isinstance(value, (dict, list)):
#                         process_dict(value)
            
#             elif isinstance(obj, list):
#                 for item in obj:
#                     if isinstance(item, (dict, list)):
#                         process_dict(item)
        
#         process_dict(query_body)

#     def _replace_with_embedding(self, knn_query: dict) -> None:
#         """Replace query_vector_builder with actual embedding vector."""
#         try:
#             model_text = (
#                 knn_query["query_vector_builder"]["text_embedding"]["model_text"]
#             )

#             # Generate embedding vector from Bedrock
#             logger.info(f"Generating Titan embedding for text: {model_text}")
#             embedding_response = self.bedrock.invoke_model(
#                 model_id="amazon.titan-embed-text-v2:0",
#                 input_text=model_text
#             )

#             # Extract embedding vector
#             if isinstance(embedding_response, dict) and "embedding" in embedding_response:
#                 embedding_vector = embedding_response["embedding"]
#             else:
#                 embedding_vector = embedding_response  # fallback if it's already a list

#             # Replace query_vector_builder with actual numeric vector
#             knn_query.pop("query_vector_builder", None)
#             knn_query["query_vector"] = embedding_vector

#             logger.info(f"Successfully replaced query_vector_builder with {len(embedding_vector)}-dim vector")
#         except Exception as e:
#             logger.error(f"Failed to generate embedding: {e}", exc_info=True)
#             raise  # Re-raise to be caught by generate_query

#     def _default_query(self) -> dict:
#         """Return a fallback match_all query."""
#         return {"query": {"match_all": {}}}





























# import json
# import re
# from src.api.routes.bedrock_client import BedrockClient, BedrockConfig
# from src.api.routes.logger import get_logger
# from src.api.routes.concern_risk_misc_naics import concerns_events, emerging_risks, misc_topics, naics_data
# from src.api.routes.settings import BEDROCK_MODEL

# logger = get_logger(__name__)

# # ------------------ OPENSEARCH SCHEMA ------------------

# OPENSEARCH_SCHEMA = """
# Index Name: ei_articles_index-31-oct

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
# - source_meta (object: {rss_entry (text), title (text)})
# - chunk_id (integer)
# - field (text)
# - chunk_text (text)
# - chunk_vector (knn_vector, 1024-dim)

# Vector Fields:
# - chunk_vector (semantic similarity search)
# """

# # ------------------ PROMPT TEMPLATE ------------------

# QUERY_GENERATION_PROMPT = """
# You are an expert at converting natural language queries into valid OpenSearch DSL queries.

# INDEX SCHEMA:
# {schema}

# USER QUERY: {query}

# RULES:
# 1. Return ONLY a valid JSON query.
# 2. Use field names exactly as shown in the schema.
# 3. Use 'knn' on 'chunk_vector' for semantic similarity.
# 4. Combine metadata filters (tag, source, region, etc.) using bool queries.
# 5. Do not add explanations or markdown, only the JSON.
# """

# # ------------------ QUERY GENERATOR CLASS ------------------

# class OpenSearchQueryGenerator:
#     def __init__(self):
#         self.bedrock = BedrockClient(BedrockConfig())
#         self.model_id = BEDROCK_MODEL
#         self.embedding_model_id = "amazon.titan-embed-text-v2:0"

#     # ---------- PREPARE SCHEMA ----------
#     def _prepare_schema(self) -> str:
#         return OPENSEARCH_SCHEMA.format(
#             concerns=", ".join(concerns_events[:15]) + "...",
#             emerging_risks=", ".join(emerging_risks[:15]) + "...",
#             misc_topics=", ".join(misc_topics[:10]) + "...",
#             naics_data=", ".join([f"{item['code']}" for item in naics_data[:10]]) + "..."
#         )

#     # ---------- GENERATE QUERY ----------
#     def generate_query(self, user_query: str) -> dict:
#         try:
#             schema = self._prepare_schema()
#             prompt = QUERY_GENERATION_PROMPT.format(schema=schema, query=user_query)

#             # Step 1: Generate query structure via LLM
#             response_text = self.bedrock.invoke_model(
#                 model_id=self.model_id,
#                 prompt=prompt,
#                 max_tokens=2000,
#                 temperature=0.0
#             )
#             logger.info(f"LLM Response (first 500 chars): {response_text[:500]}")

#             query_body = self._extract_json(response_text)
#             if not query_body:
#                 logger.warning("Empty or invalid JSON from LLM, using match_all")
#                 return self._default_query()

#             # Step 2: Replace text embedding blocks with real Titan embeddings
#             query_body = self._replace_text_embeddings(query_body)

#             logger.info(f"✅ Final query ready for OpenSearch:\n{json.dumps(query_body, indent=2)}")
#             return query_body

#         except Exception as e:
#             logger.error(f"Error generating OpenSearch query: {e}", exc_info=True)
#             return self._default_query()

#     # ---------- REPLACE TEXT EMBEDDINGS ----------
#     def _replace_text_embeddings(self, query_body: dict) -> dict:
#         """Find and replace 'query_vector_builder' blocks with real Titan embeddings."""
#         try:
#             if not query_body:
#                 return query_body

#             def traverse(obj):
#                 """Recursively search for knn blocks in nested structures."""
#                 if isinstance(obj, dict):
#                     if "knn" in obj:
#                         knn_query = obj["knn"]
#                         if "query_vector_builder" in knn_query:
#                             text_to_embed = (
#                                 knn_query["query_vector_builder"]
#                                 .get("text_embedding", {})
#                                 .get("model_text")
#                             )
#                             if text_to_embed:
#                                 logger.info(f"Generating Titan embedding for: {text_to_embed}")
#                                 embed_response = self.bedrock.invoke_model(
#                                     model_id=self.embedding_model_id,
#                                     input_text=text_to_embed
#                                 )

#                                 # Titan may return either a list or dict
#                                 if isinstance(embed_response, dict):
#                                     embedding_vector = (
#                                         embed_response.get("embedding")
#                                         or embed_response.get("embeddingVector")
#                                         or embed_response.get("vector")
#                                     )
#                                 else:
#                                     embedding_vector = embed_response

#                                 if not isinstance(embedding_vector, list):
#                                     raise ValueError("Titan embedding is not a list")

#                                 knn_query.pop("query_vector_builder", None)
#                                 knn_query["query_vector"] = embedding_vector
#                                 logger.info("✅ Replaced text_embedding with numeric vector")

#                     # Continue traversal
#                     for k, v in obj.items():
#                         traverse(v)
#                 elif isinstance(obj, list):
#                     for item in obj:
#                         traverse(item)

#             traverse(query_body)
#             return query_body

#         except Exception as e:
#             logger.error(f"Error while replacing text embeddings: {e}", exc_info=True)
#             return query_body

#     # ---------- EXTRACT JSON ----------
#     def _extract_json(self, text: str) -> dict:
#         text = text.strip()
#         if "```" in text:
#             match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
#             if match:
#                 text = match.group(1).strip()

#         try:
#             return json.loads(text)
#         except json.JSONDecodeError:
#             # Try to extract valid JSON part manually
#             start = text.find("{")
#             end = text.rfind("}") + 1
#             if start >= 0 and end > start:
#                 try:
#                     return json.loads(text[start:end])
#                 except Exception as e:
#                     logger.error(f"Failed JSON parse after extraction: {e}")
#         logger.warning("No valid JSON found in LLM output")
#         return {}

#     # ---------- DEFAULT QUERY ----------
#     def _default_query(self) -> dict:
#         return {"query": {"match_all": {}}}





























# import json
# import re
# from typing import Optional
# from src.api.routes.bedrock_client import BedrockClient, BedrockConfig
# from src.api.routes.logger import get_logger
# from src.api.routes.concern_risk_misc_naics import concerns_events, emerging_risks, misc_topics, naics_data
# from src.api.routes.settings import BEDROCK_MODEL

# logger = get_logger(__name__)

# # ------------------ OPENSEARCH SCHEMA ------------------

# OPENSEARCH_SCHEMA = """ 
# Index Name: ei_articles_index-05-nov-test

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

# # ------------------ PROMPT TEMPLATE ------------------

# QUERY_GENERATION_PROMPT = """
# You are an expert at converting natural language queries into valid OpenSearch DSL queries.

# INDEX SCHEMA:
# {schema}

# USER QUERY: {query}

# CRITICAL INSTRUCTIONS:
# 1. Generate ONLY a valid JSON object for the OpenSearch query body.
# 2. Use EXACT field names from the schema (case-sensitive).
# 3. Query rules:
#    - If the query involves *content*, *title*, *reason*, or any semantic topic (e.g. mentions data, reason_identified, title, or general article topic):
#        → Use a similarity (vector) search on `chunk_vector` with a KNN query.
#        Example:
#        {{
#          "knn": {{
#            "field": "chunk_vector",
#            "query_vector_builder": {{
#              "text_embedding": {{
#                "model_id": "embedding_model_id",
#                "model_text": "<USER_QUERY_TEXT>"
#              }}
#            }},
#            "k": 10,
#            "num_candidates": 100
#          }}
#        }}
#    - For metadata searches (e.g. tag, source, region, concerns, emerging_risk_name, miscTopics, naicscode):
#        → Use `term`, `terms`, `match`, or `range` queries on those fields.
#    - To combine similarity and filters (hybrid search):
#        → Use a `bool` query where `must` includes both the `knn` and any filters.
#        ✅ Do NOT place `knn` inside `filter`; it must be directly under `must`.

# 4. Field types:
#    - Keyword: tag, source, concerns, emerging_risk_name, miscTopics, naicscode, naics_description, region, url
#    - Text: title, description, data, reason_identified, chunk_text
#    - Boolean: is_latest
#    - Numeric: doc_id, chunk_id
#    - Date/time: published_time, last_update_time, injection_time

# 5. Common query patterns:
#    - "Show all articles" → match_all
#    - "Tagged as Current" → term query on tag
#    - "From Reuters" → term query on source
#    - "With concern PFAS" → term query on concerns
#    - "Show recent articles" → range query on published_time or last_update_time

# 6. Combination examples:
#    - "Articles about climate change tagged as Current":
#      → Use `bool.must` with knn query for "climate change" and a term query for tag="Current".
#    - "Emerging risks in Europe":
#      → term on region="Europe" + knn if context indicates semantic topic.
#    - "Show PFAS untagged articles":
#      → bool.must with term(tag="Untagged") and term(concerns="PFAS").
#    - "Articles about wildfires in last 3 days":
#      → bool.must with range on published_time and knn for "wildfires".

# 7. Always structure hybrid queries like this:
#    {{
#      "query": {{
#        "bool": {{
#          "must": [
#            <other field filters>,
#            {{
#              "knn": {{
#                "field": "chunk_vector",
#                "query_vector_builder": {{
#                  "text_embedding": {{
#                    "model_id": "embedding_model_id",
#                    "model_text": "<USER_QUERY_TEXT>"
#                  }}
#                }},
#                "k": 10,
#                "num_candidates": 100
#              }}
#            }}
#          ]
#        }}
#      }}
#    }}

# 8. Return ONLY the JSON object — no text, markdown, or explanations.

# ---

# EXAMPLES:

# Query: "Show me all articles tagged as Current"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Current"
#     }}
#   }}
# }}

# Query: "Find articles about climate change"
# {{
#   "knn": {{
#     "field": "chunk_vector",
#     "query_vector_builder": {{
#       "text_embedding": {{
#         "model_id": "embedding_model_id",
#         "model_text": "climate change"
#       }}
#     }},
#     "k": 10,
#     "num_candidates": 100
#   }}
# }}

# Query: "Find articles about lawsuits tagged as Current"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "term": {{
#             "tag": "Current"
#           }}
#         }},
#         {{
#           "knn": {{
#             "field": "chunk_vector",
#             "query_vector_builder": {{
#               "text_embedding": {{
#                 "model_id": "embedding_model_id",
#                 "model_text": "lawsuits"
#               }}
#             }},
#             "k": 10,
#             "num_candidates": 100
#           }}
#         }}
#       ]
#     }}
#   }}
# }}

# Query: "Show untagged articles"
# {{
#   "query": {{
#     "term": {{
#       "tag": "Untagged"
#     }}
#   }}
# }}

# Query: "Show articles with wildfire for last 3 days"
# {{
#   "query": {{
#     "bool": {{
#       "must": [
#         {{
#           "range": {{
#             "published_time": {{
#               "gte": "now-3d/d",
#               "lte": "now"
#             }}
#           }}
#         }},
#         {{
#           "knn": {{
#             "field": "chunk_vector",
#             "query_vector_builder": {{
#               "text_embedding": {{
#                 "model_id": "embedding_model_id",
#                 "model_text": "wildfire"
#               }}
#             }},
#             "k": 10,
#             "num_candidates": 100
#           }}
#         }}
#       ]
#     }}
#   }}
# }}
# """


# # ------------------ QUERY GENERATOR CLASS ------------------
# # Query: "injury"
# # {{
# #   "query": {{
# #     "multi_match": {{
# #       "query": "injury",
# #       "fields": ["title", "description", "data", "reason_identified"]
# #     }}
# #   }}
# # }}



# class OpenSearchQueryGenerator:
#     def __init__(self):
#         self.bedrock = BedrockClient(BedrockConfig())
#         self.model_id = BEDROCK_MODEL

#     def _prepare_schema(self) -> str:
#         """Prepare schema for the prompt."""
#         return OPENSEARCH_SCHEMA.format(
#             concerns=", ".join(concerns_events) + "...",
#             emerging_risks=", ".join(emerging_risks) + "...",
#             misc_topics=", ".join(misc_topics) + "...",
#             naics_data=", ".join([f"{item['code']}" for item in naics_data]) + "..."
#         )

#     def generate_query(self, user_query: str, source_filter: Optional[str] = None) -> dict:
#         """Generate an OpenSearch query DSL body from a natural language query."""
#         try:
#             embedding_model_id = "amazon.titan-embed-text-v2:0"
#             schema = self._prepare_schema()
#             prompt = QUERY_GENERATION_PROMPT.format(schema=schema, query=user_query)
#             prompt = prompt.replace("embedding_model_id", embedding_model_id)

#             response_text = self.bedrock.invoke_model(
#                 model_id=self.model_id,
#                 prompt=prompt,
#                 max_tokens=2000,
#                 temperature=0.0
#             )

#             logger.info(f"LLM Response (first 500 chars): {response_text[:500]}...")
            
#             query_body = self._extract_json(response_text)

#             if not query_body or ("query" not in query_body and "knn" not in query_body):
#                 logger.warning(f"Invalid query generated, using match_all. Response: {response_text[:200]}")
#                 query_body = self._default_query()

#             # Ensure query_body has a query structure
#             if "query" not in query_body:
#                 query_body["query"] = {"match_all": {}}

#             if "knn" in query_body:
#               knn_query = query_body["knn"]

#               # Check if LLM used query_vector_builder
#               if "query_vector_builder" in knn_query:
#                   try:
#                       model_text = (
#                           knn_query["query_vector_builder"]["text_embedding"]["model_text"]
#                       )

#                       # Generate embedding vector from Bedrock
#                       logger.info(f"Generating Titan embedding for text: {model_text}")
#                       embedding_response = self.bedrock.invoke_model(
#                           model_id="amazon.titan-embed-text-v2:0",
#                           input_text=model_text
#                       )

#                       # Expect embedding_response["embedding"] or similar, adjust if needed
#                       if isinstance(embedding_response, dict) and "embedding" in embedding_response:
#                           embedding_vector = embedding_response["embedding"]
#                       else:
#                           embedding_vector = embedding_response  # fallback if it's already a list

#                       # Replace query_vector_builder with actual numeric vector
#                       knn_query.pop("query_vector_builder", None)
#                       knn_query["query_vector"] = embedding_vector

#                       logger.info("Successfully replaced query_vector_builder with numeric vector")
#                   except Exception as e:
#                       logger.error(f"Failed to generate embedding: {e}")

#             # if "knn" in query_body:
#             #   logger.info("Generated vector similarity (semantic) search query.")
#             # else:
#             #   logger.info("Generated field-based query.")

#             # Add source filter if provided
#             if source_filter:
#                 original_query = query_body["query"]
                
#                 # Check if original query is already a bool query
#                 if isinstance(original_query, dict) and "bool" in original_query:
#                     # Merge into existing bool query
#                     bool_query = original_query["bool"]
                    
#                     if source_filter == "other":
#                         # Exclude CourtListener - add to must_not
#                         if "must_not" not in bool_query:
#                             bool_query["must_not"] = []
#                         bool_query["must_not"].append({"term": {"source": "court_listener"}})
#                     else:
#                         # Filter for specific source - add to must
#                         if "must" not in bool_query:
#                             bool_query["must"] = []
#                         bool_query["must"].append({"term": {"source": source_filter}})
#                 else:
#                     # Original query is not a bool query - wrap it to preserve the user's query
#                     if source_filter == "other":
#                         # Exclude CourtListener - wrap user query with must_not
#                         query_body["query"] = {
#                             "bool": {
#                                 "must": [original_query],
#                                 "must_not": [{"term": {"source": "court_listener"}}]
#                             }
#                         }
#                     else:
#                         # Filter for specific source - wrap user query with source filter
#                         query_body["query"] = {
#                             "bool": {
#                                 "must": [
#                                     original_query,
#                                     {"term": {"source": source_filter}}
#                                 ]
#                             }
#                         }

#             logger.info(f"Successfully generated OpenSearch query for: {user_query}")
#             if source_filter:
#                 logger.info(f"Source filter applied: {source_filter}")
#             logger.info(f"Query body: {json.dumps(query_body, indent=2)}")
#             return query_body

#         except Exception as e:
#             logger.error(f"Error generating OpenSearch query: {e}", exc_info=True)
#             return self._default_query()

#     def _extract_json(self, text: str) -> dict:
#         """Extract JSON from the LLM's response text with robust error handling."""
#         # Remove common markdown artifacts
#         text = text.strip()
        
#         # Remove markdown code blocks if present
#         if '```' in text:
#             # Extract content between code blocks
#             match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
#             if match:
#                 text = match.group(1).strip()
        
#         try:
#             # Try direct parsing
#             return json.loads(text)
#         except json.JSONDecodeError as e:
#             logger.warning(f"Direct JSON parse failed: {e}")
            
#             try:
#                 # Find the first complete JSON object
#                 start = text.find('{')
#                 if start == -1:
#                     logger.error(f"No opening brace found in: {text[:100]}")
#                     return {}
                
#                 # Count braces to find matching closing brace
#                 brace_count = 0
#                 in_string = False
#                 escape_next = False
#                 end = start
                
#                 for i in range(start, len(text)):
#                     char = text[i]
                    
#                     if escape_next:
#                         escape_next = False
#                         continue
                    
#                     if char == '\\':
#                         escape_next = True
#                         continue
                    
#                     if char == '"':
#                         in_string = not in_string
#                         continue
                    
#                     if not in_string:
#                         if char == '{':
#                             brace_count += 1
#                         elif char == '}':
#                             brace_count -= 1
#                             if brace_count == 0:
#                                 end = i + 1
#                                 break
                
#                 if end > start:
#                     json_str = text[start:end]
#                     parsed = json.loads(json_str)
#                     logger.info("Successfully extracted JSON from text")
#                     return parsed
#                 else:
#                     logger.error("Could not find matching closing brace")
                    
#             except json.JSONDecodeError as e2:
#                 logger.error(f"Failed to extract valid JSON: {e2}")
#                 logger.error(f"Attempted to parse: {text[start:min(start+200, len(text))]}...")
#             except Exception as e3:
#                 logger.error(f"Unexpected error during JSON extraction: {e3}")
        
#         logger.error("All JSON extraction attempts failed, returning empty dict")
#         return {}

#     def _default_query(self) -> dict:
#         """Return a fallback match_all query."""
#         return {"query": {"match_all": {}}}



