import asyncio
# from CommonService.async_bedrock import Embedding
from CommonService.async_bedrock import TitanV1, TitanV2
from CommonService.async_commonsession.commonsession import (
    CommonSession,
    CommonSessionConfig,
)


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
    asyncio.run(embed())
