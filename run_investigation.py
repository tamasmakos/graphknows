import asyncio, sys, os, json, re, math, random
sys.path.insert(0, '/root/graphknows/services')
sys.path.insert(0, '/root/graphknows/services/agent')
sys.path.insert(0, '/root/graphknows/services/graphgen')

from dotenv import load_dotenv
load_dotenv('/root/graphknows/.env')

from agent.langgraph_agent import build_graph
from langchain_core.messages import HumanMessage

collected_thoughts = []
collected_answer = []
tool_calls = []

async def broadcaster(session_id, payload):
    t = payload.get('type','')
    if t == 'thought_token':
        collected_thoughts.append(payload.get('token',''))
    elif t == 'answer_token':
        collected_answer.append(payload.get('token',''))
    elif t == 'tool_call':
        tool_calls.append(payload)

async def main():
    graph = build_graph(
        session_id='ecb-cbdc-001',
        groq_api_key=os.environ['GROQ_API_KEY'],
        ws_broadcaster=broadcaster,
    )

    result = await graph.ainvoke(
        {'messages': [HumanMessage(content='Investigate the ECB digital euro CBDC project: institutional actors, legislative timeline, political opposition, and monetary sovereignty concerns. Write a full EPRS-style intelligence brief for the ECON committee.')]},
    )

    answer = ''.join(collected_answer) or (result['messages'][-1].content if result['messages'] else '')
    
    # Save answer to file
    with open('/root/graphknows/output/agent_answer.txt', 'w') as f:
        f.write(answer)
    
    with open('/root/graphknows/output/tool_calls.json', 'w') as f:
        json.dump(tool_calls, f, default=str)
    
    print('=== ANSWER (first 500 chars) ===')
    print(answer[:500])
    print('=== ANSWER LENGTH ===')
    print(len(answer))

asyncio.run(main())
