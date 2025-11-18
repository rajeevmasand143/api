

import pandas as pd

from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

import boto3

from botocore.config import Config

import json

import httpx

import json

from collections import Counter
 
 
PORT=443

OPENSEARCH_HOST="tv9xe9sa7lpqtaqr5o9k.us-east-1.aoss.amazonaws.com"

SERVICE = "aoss"

AWS_PROFILE_NAME = "sandbox"

region = "us-east-1"

credentials = boto3.Session().get_credentials()

auth = AWSV4SignerAuth(credentials, region, SERVICE)
 
 
os_client = OpenSearch(

                hosts=[{"host":OPENSEARCH_HOST , "port": PORT}],

                http_auth=auth,

                use_ssl=True,

                verify_certs=True,

                connection_class=RequestsHttpConnection,

                pool_maxsize=20,

)

query = {

  "query": {

    "bool": {

      "must": [

        { "match": { "lob": "GL" } },

          {"exists": { "field": "suitable_state.AZ" }},

          {"match_phrase": {"file_path": "/datahub-clm/SIM/GL/AZ/LC/R_0.504_GL-AZ-2026-LC-001.htm"}}

      ]

    }

  },

"_source": {

    "excludes": ["vector_field"]

  },

    "size": 2000

}
 
 
response = os_client.search(index="manuals_gl_v2", body=query)

for hit in response['hits']['hits']:

    print(hit['_source']['documentName'], hit['_source']['file_path'])
 