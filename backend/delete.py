# Quick delete script - save as delete_case.py
import os
from azure.cosmos import CosmosClient
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

cosmos_client = CosmosClient(
    os.getenv("COSMOS_ENDPOINT"),
    os.getenv("COSMOS_KEY")
)
database = cosmos_client.get_database_client(os.getenv("DATABASE_NAME"))
container = database.get_container_client(os.getenv("CASES_CONTAINER_NAME"))

# Find case
query = "SELECT * FROM c WHERE c.caseNumber = 'SR-P-NAV' AND c.type = 'case'"

cases = list(container.query_items(query=query, enable_cross_partition_query=True))

if cases:
    case = cases[0]
    container.delete_item(item=case['id'], partition_key=case['userId'])
    print(f"✅ Deleted case: {case['caseNumber']}")
else:
    print("Case not found")

query = "SELECT * FROM c WHERE c.caseNumber = 'MX-B-SAL' AND c.type = 'case'"
cases = list(container.query_items(query=query, enable_cross_partition_query=True))

if cases:
    case = cases[0]
    container.delete_item(item=case['id'], partition_key=case['userId'])
    print(f"✅ Deleted case: {case['caseNumber']}")
else:
    print("Case not found")
