from fastapi import APIRouter, Depends, Request
from opensearchpy import AsyncOpenSearch

from src.core.config_loader import get_opensearch_client
from src.logger.console_logs import Loggercheck
from CommonService.async_opensearch.service import dependency, lifespan_factory
from ...utils.utils import merge_with_overlap
from src.api.routes.query_generator import OpenSearchQueryGenerator
# from src.aconcern_risk_misc_naics import concerns_events, emerging_risks, misc_topics, naics_data
from src.api.routes.search_opensearch import search_documents, get_unique_docs
import json

logger_instance = Loggercheck(__name__)
logger = logger_instance.get_logger()

sample_router = APIRouter(prefix="/v1", tags=["v1"])


@sample_router.get("/mappings")
async def get_indexes(request: Request, client: AsyncOpenSearch = Depends(dependency())):
    session = request.state.session
    index = "ei_articles_index"
    logger_instance.logg_message(
        f"{session} - Searching on index - {index}",
        "info",
    )
    try:
        res = await client.indices.get_mapping(index=index)
        return res
    except Exception as e:
        logger_instance.logg_message(
            f"{session} - Error fetching the result - {e}",
            "info",
        )

        return {"error": str(e)}
    
@sample_router.get("/doc-search")
async def get_indexes(request: Request, client: AsyncOpenSearch = Depends(dependency())):
    session = request.state.session
    try:
        # res = await client.indices.get_mapping(index=index)
        response = await client.search(body={
                        "query": {"match_all": {}},
                        "size": 1,
                    }, index="ei_articles_index")
        return response
    except Exception as e:
        logger_instance.logg_message(
            f"{session} - Error fetching the result - {e}",
            "info",
        )

        return {"error": str(e)}


@sample_router.get("/url-search")
async def get_indexes(request: Request, client: AsyncOpenSearch = Depends(dependency())):
    session = request.state.session
    try:
        # res = await client.indices.get_mapping(index=index)
        url = "https://techxplore.com/news/2025-10-uber-partners-nvidia-deploy-robotaxis.html"
        search_query = {
            "query": {
                "term": {
                    "URL":  { 
                        "value":f"{url}"  
                    }
                }
            }
        }
        response = await client.search(body=search_query, index="ei_articles_index")
        hits = response.get("hits", {}).get("hits", [])
        chunks = [hit["_source"]["Data"] for hit in hits if "_source" in hit and "Data" in hit["_source"]]

        # Merge all the chunks considering overlap of 150
        reconstructed_doc = merge_with_overlap(chunks, overlap=150)
        return reconstructed_doc

    except Exception as e:
        logger_instance.logg_message(
            f"{session} - Error fetching the result - {e}",
            "info",
        )

        return {"error": str(e)}
    
    
@sample_router.get("/health")
def health_check(request: Request):
    client = request.state.os_client
    try:
        health = client.cluster.health()
        if health["status"] in ["green", "yellow"]:
            return {"status": "ok", "Opensearch": health["status"]}
        else:
            return {"status": "down", "Opensearch": health["status"]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


from src.models.search_schemas import Query


@sample_router.post("/search-insights")
async def search_query(query: Query):
    query_generator = OpenSearchQueryGenerator()
    query_params = await query_generator.generate_query(query.query, source_filter=query.source)
    if not query_params:
        return {
            "message": "Query recieved successfully!",
            "user_query": query.query,
            "source_filter": query.source,
            "query_params": False,
            "results": False
        }

    query_params['size'] = 500  # Increased to get more results for grouping
    
    # Debug logging
    logger.info(f"Generated query_params: {query_params}")
    logger.info(f"Source filter: {query.source}")
    logger.info(f"User query: {query.query}")
    
    index_name = "ei_articles_index"
    print(query_params)
    # return None
    search_results= search_documents(index_name, query_params, size=500)
    # with open('search_results.json', 'w') as file:
    #     file.write(json.dumps(search_results))
    # print("search results are sankarr>>",search_results)
    get_unique_doc = get_unique_docs(search_results)
    # with open('unique_results.json', 'w') as file:
    #     file.write(json.dumps(get_unique_doc))
    
    # Log search results summary
    if isinstance(search_results, dict) and "hits" in search_results:
        total_hits = search_results.get("hits", {}).get("total", {})
        if isinstance(total_hits, dict):
            total_count = total_hits.get("value", 0)
        else:
            total_count = total_hits
        logger.info(f"Search returned {total_count} total hits")
    else:
        logger.warning(f"Unexpected search results format: {type(search_results)}")
    # print("unique docuemnts results>>>>>>dhruv>>>>>>",get_unique_docs)
    return {
        "message": "Query recieved successfully!",
        "user_query": query.query,
        "source_filter": query.source,
        "query_params": query_params,
        "results": get_unique_doc
    }



