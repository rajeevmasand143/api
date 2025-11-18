import asyncio
# from CommonService.async_bedrock import Embedding
from CommonService.async_bedrock import TitanV1, TitanV2
from CommonService.async_commonsession.commonsession import (
    CommonSession,
    CommonSessionConfig,
)

import asyncio
from opensearchpy import AsyncOpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

async def search_articles(collection, index, query):
    body = {
        "query": {
            "match": {
                "url": query
            }
        }
    }

    async with CommonSession(client_name="aoss", region="us-east-1", profile_name="") as client:
        response = await client.search(
            CollectionName=collection,
            IndexName=index,
            Body=body
        )
        return response
    

async def connect_and_index():
    region = "us-east-1"
    service = "aoss"  # OpenSearch Serverless
    collection_endpoint = "https://tv9xe9sa7lpqtaqr5o9k.us-east-1.aoss.amazonaws.com"
    index_name = "ei_articles_index"

    # Use IAM role credentials automatically provided in sandbox
    session = boto3.Session()
    credentials = session.get_credentials().get_frozen_credentials()

    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
    )

    # Connect to OpenSearch Serverless
    client = AsyncOpenSearch(
        hosts=[collection_endpoint],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    try:
        doc = {"text": "Hello world!"}
        response = await client.index(index=index_name, body=doc)
        print(response)
    finally:
        await client.transport.close()


async def embed():
    async with CommonSession(
        CommonSessionConfig(
            client_name="bedrock-runtime", region="us-east-1", profile_name=""
        )
    ) as titan:
        titan_embedding = TitanV1(titan)
        response = await titan_embedding.generate_embedding({"inputText": "Hello world!"})
        print(response)

if __name__ == "__main__":
    # asyncio.run(embed())
    asyncio.run(connect_and_index())
    # asyncio.run(search_articles('https://tv9xe9sa7lpqtaqr5o9k.us-east-1.aoss.amazonaws.com','ei_articles_index','https://storage.courtlistener.com/recap/gov.uscourts.mied.388874/gov.uscourts.mied.388874.1.0.pdf'))
