import os
from typing import List, Tuple

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings


# ============================================================
# 1. 환경 변수 로드
# ============================================================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


# ============================================================
# 2. 기존 Neo4j 인덱스 이름
# ============================================================
#
# chunk_vector_index:
#   (:Chunk).embedding 속성을 대상으로 만든 VECTOR INDEX
#
# chunk_keyword_index:
#   (:Chunk).text 속성을 대상으로 만든 FULLTEXT INDEX
# ============================================================

VECTOR_INDEX_NAME = "chunk_vector_index"
KEYWORD_INDEX_NAME = "chunk_keyword_index"


# ============================================================
# 3. 그래프 확장 범위
# ============================================================
#
# 이 예제의 retrieval_query는 다음 범위로 그래프를 확장합니다.
#
# 검색된 Chunk
#   ├─ 이전 Chunk 1개
#   ├─ 다음 Chunk 1개
#   ├─ MENTIONS 관계로 연결된 KGEntity 최대 20개
#   └─ 각 KGEntity와 연결된 다른 KGEntity 관계 최대 40개
# ============================================================

MAX_ENTITIES_PER_CHUNK = 20
MAX_ENTITY_RELATIONSHIPS_PER_CHUNK = 40


# ============================================================
# 4. 필수 환경 변수 검사
# ============================================================

def validate_environment() -> None:
    required_values = {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    }

    missing = [
        key
        for key, value in required_values.items()
        if value is None or not str(value).strip()
    ]

    if missing:
        raise ValueError(
            "다음 환경 변수가 설정되지 않았습니다: "
            + ", ".join(missing)
        )


validate_environment()


# ============================================================
# 5. Neo4j 연결
# ============================================================
#
# Neo4jGraph:
#   - 연결 상태 확인
#   - 인덱스 상태 확인
#   - 그래프 데이터 개수 확인
#
# Neo4jVector:
#   - 실제 Vector + Keyword Hybrid Search
#   - 검색된 Chunk를 시작점으로 retrieval_query 실행
# ============================================================

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)


# ============================================================
# 6. Embedding과 LLM 준비
# ============================================================
#
# 중요:
# 기존 Chunk.embedding 생성에 사용한 embedding 모델과
# 여기서 사용하는 모델이 달라지면 벡터 차원 또는 의미 공간이
# 달라져 올바른 검색이 되지 않음
#
# llm: Hybrid GraphRAG로 가져온 문맥만 사용해 최종 답변 생성
# ============================================================

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
)

llm = init_chat_model(
    model=OPENAI_MODEL,
    model_provider="openai",
)


# ============================================================
# 7. Hybrid GraphRAG Expansion Query
# ============================================================
#
# Neo4jVector의 Hybrid Search가 먼저 실행된 뒤,
# 검색된 각 Chunk가 아래 쿼리의 node 변수로 전달
#
# Neo4jVector가 retrieval_query에 제공하는 변수:
#   node  : Hybrid Search에서 검색된 Chunk 노드
#   score : 해당 검색 결과의 점수
#
# 이 쿼리에서 수행하는 Graph Expansion:
#
# 1) 이전 Chunk
#    (previous:Chunk)-[:NEXT_CHUNK]->(node)
#
# 2) 다음 Chunk
#    (node)-[:NEXT_CHUNK]->(next:Chunk)
#
# 3) 현재 Chunk가 언급한 Entity
#    (node)-[:MENTIONS]->(entity:KGEntity)
#
# 4) Entity 주변의 1-hop 관계
#    (entity)-[rel]-(related:KGEntity)
#
# ============================================================

retrieval_query = f"""
// ------------------------------------------------------------
// 1. 검색된 Chunk의 이전 Chunk 수집
// ------------------------------------------------------------
OPTIONAL MATCH (previous:Chunk)-[:NEXT_CHUNK]->(node)

WITH
    node,
    score,
    collect(
        DISTINCT CASE
            WHEN previous IS NULL THEN null
            ELSE {{
                chunk_id: previous.id,
                text: previous.text,
                page_number: previous.page_number,
                source: previous.source
            }}
        END
    )[0..1] AS previous_chunks


// ------------------------------------------------------------
// 2. 검색된 Chunk의 다음 Chunk 수집
// ------------------------------------------------------------
OPTIONAL MATCH (node)-[:NEXT_CHUNK]->(next:Chunk)

WITH
    node,
    score,
    previous_chunks,
    collect(
        DISTINCT CASE
            WHEN next IS NULL THEN null
            ELSE {{
                chunk_id: next.id,
                text: next.text,
                page_number: next.page_number,
                source: next.source
            }}
        END
    )[0..1] AS next_chunks


// ------------------------------------------------------------
// 3. 검색된 Chunk가 언급한 KGEntity 수집
// ------------------------------------------------------------
OPTIONAL MATCH (node)-[:MENTIONS]->(entity:KGEntity)

WITH
    node,
    score,
    previous_chunks,
    next_chunks,
    collect(DISTINCT entity)[0..{MAX_ENTITIES_PER_CHUNK}]
        AS entity_nodes


// ------------------------------------------------------------
// 4. Entity가 없는 경우에도 검색된 Chunk가 사라지지 않도록
//    null 한 개를 가진 리스트로 바꾸어 UNWIND
// ------------------------------------------------------------
UNWIND
    CASE
        WHEN size(entity_nodes) = 0 THEN [null]
        ELSE entity_nodes
    END AS entity


// ------------------------------------------------------------
// 5. 각 Entity 주변의 1-hop KGEntity 관계 탐색
//
// 방향을 양방향으로 탐색하되,
// relationship_direction 필드에 실제 관계 방향을 기록합니다.
//
// 예:
//   (보장)-[:REQUIRES_DOCUMENT]->(청구서류)
//
// anchor_entity가 보장이라면 OUTGOING,
// anchor_entity가 청구서류라면 INCOMING으로 기록됩니다.
// ------------------------------------------------------------
OPTIONAL MATCH (entity)-[rel]-(related:KGEntity)

WITH
    node,
    score,
    previous_chunks,
    next_chunks,
    entity_nodes,
    collect(
        DISTINCT CASE
            WHEN rel IS NULL OR related IS NULL THEN null
            ELSE {{
                anchor_entity: entity.name,
                anchor_entity_type: entity.type,

                relationship: type(rel),

                relationship_direction:
                    CASE
                        WHEN startNode(rel) = entity THEN "OUTGOING"
                        ELSE "INCOMING"
                    END,

                relationship_properties: properties(rel),

                related_entity: related.name,
                related_entity_type: related.type,
                related_entity_description: related.description
            }}
        END
    )[0..{MAX_ENTITY_RELATIONSHIPS_PER_CHUNK}]
        AS entity_relationships


// ------------------------------------------------------------
// 6. Neo4jVector 반환 형식
//
// text:
//   Document.page_content가 되는 검색 문맥
//
// score:
//   Hybrid Search 점수
//
// metadata:
//   Document.metadata가 되는 출처 정보
// ------------------------------------------------------------
RETURN
    {{
        matched_chunk: {{
            chunk_id: node.id,
            text: node.text,
            page_number: node.page_number,
            source: node.source
        }},

        previous_chunks: previous_chunks,
        next_chunks: next_chunks,

        mentioned_entities: [
            item IN entity_nodes |
            {{
                name: item.name,
                type: item.type,
                description: item.description
            }}
        ],

        entity_relationships: entity_relationships
    }} AS text,

    score,

    {{
        chunk_id: node.id,
        page_number: node.page_number,
        source: node.source
    }} AS metadata
"""


# ============================================================
# 8. 기존 Hybrid Index를 사용하는 Neo4jVector 생성
# ============================================================

vector_store = Neo4jVector.from_existing_index(
    embedding=embeddings,

    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,

    index_name=VECTOR_INDEX_NAME,
    keyword_index_name=KEYWORD_INDEX_NAME,

    search_type="hybrid",
    retrieval_query=retrieval_query,
)


# ============================================================
# 9. Hybrid GraphRAG 검색 함수
# ============================================================
#
# 내부 처리 순서:
#
# 1. 사용자 질문 Embedding 생성
# 2. Vector Index 검색
# 3. Full-text Keyword Index 검색
# 4. Hybrid 결과 병합
# 5. 각 검색 Chunk를 node로 retrieval_query 실행
# 6. 확장된 Graph Context를 Document로 반환

# ============================================================

SearchResult = Tuple[Document, float]


def search_hybrid_graphrag(
    question: str,
    k: int = 5,
) -> List[SearchResult]:
    if not question or not question.strip():
        raise ValueError("질문을 입력해야 합니다.")

    if k < 1:
        raise ValueError("k는 1 이상의 정수여야 합니다.")

    return vector_store.similarity_search_with_score(
        query=question,
        k=k,
    )


# ============================================================
# 10. 검색 결과를 LLM Context로 변환
# ============================================================
#
# retrieval_query의 반환값 : 
# text     -> doc.page_content
# metadata -> doc.metadata
# score    -> similarity_search_with_score()의 score
#
# Graph Expansion 결과가 각 검색 결과 단위로 구분되도록
# 제목과 메타데이터를 붙여 하나의 context 문자열로 병합
# ============================================================

def format_context(
    results: List[SearchResult],
) -> str:
    if not results:
        return "검색된 문맥이 없습니다."

    formatted_results = []

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):
        metadata = document.metadata or {}

        source = metadata.get("source", "출처 정보 없음")
        page_number = metadata.get(
            "page_number",
            "페이지 정보 없음",
        )
        chunk_id = metadata.get(
            "chunk_id",
            "Chunk ID 없음",
        )

        formatted_results.append(
            f"""
[검색 결과 {index}]
Hybrid 검색 점수:
{score}

출처:
- source: {source}
- page_number: {page_number}
- chunk_id: {chunk_id}

검색된 Chunk와 확장된 Graph Context:
{document.page_content}
""".strip()
        )

    return "\n\n".join(formatted_results)


# ============================================================
# 11. 답변 생성 프롬프트
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
너는 보험 약관 기반 Hybrid GraphRAG assistant다.

제공되는 context에는 다음 정보가 포함될 수 있다.

1. Hybrid Search로 직접 검색된 matched_chunk
2. matched_chunk의 previous_chunks
3. matched_chunk의 next_chunks
4. matched_chunk가 언급한 mentioned_entities
5. Entity에서 1-hop 확장한 entity_relationships

답변 규칙:

- 반드시 제공된 context에 근거해서만 답변하라.
- context에 없는 보장 내용, 면책 조건, 금액, 기간, 청구 절차를
  임의로 만들어내지 마라.
- 사용자가 일상적인 표현을 사용하더라도 보험 약관의 공식 용어로
  연결하여 설명하라.
- matched_chunk의 원문 내용을 가장 우선적인 근거로 사용하라.
- previous_chunks와 next_chunks는 문맥 보충에 사용하라.
- entity_relationships는 보장, 면책, 조건, 필요 서류,
  처리 절차 사이의 구조를 이해하는 보조 근거로 사용하라.
- Chunk 내용과 Entity 관계가 충돌하면 Chunk 원문을 우선하라.
- 확실한 근거가 없으면 확인할 수 없다고 명시하라.
- 답변 마지막에는 근거로 사용한 source와 page_number를 적어라.
- 검색 결과가 질문에 충분하지 않으면 추가로 확인해야 할 사항을
  간단히 안내하라.
""".strip(),
        ),
        (
            "human",
            """
질문:
{question}

검색된 Hybrid GraphRAG context:
{context}
""".strip(),
        ),
    ]
)


# ============================================================
# 12. 최종 질의응답 함수
# ============================================================
#
# 처리 순서:
#
# 사용자 질문
#   ↓
# Vector Search + Keyword Search
#   ↓
# 상위 k개 Chunk
#   ↓
# 이전/다음 Chunk 확장
#   ↓
# MENTIONS Entity 확장
#   ↓
# Entity 주변 1-hop 관계 확장
#   ↓
# LLM Context 구성
#   ↓
# 최종 답변
# ============================================================

def answer_question(
    question: str,
    k: int = 5,
    show_context: bool = False,
) -> str:
    # 1. Hybrid Search + Graph Expansion
    search_results = search_hybrid_graphrag(
        question=question,
        k=k,
    )

    # 2. LLM에 전달할 문자열 Context 생성
    context = format_context(search_results)

    # 3. 검색 결과 확인이 필요할 때 출력
    if show_context:
        print("\n")
        print("[LLM에 전달되는 Hybrid GraphRAG Context]")
        print(context)

    # 4. Prompt에 질문과 Context 주입
    messages = prompt.format_messages(
        question=question,
        context=context,
    )

    # 5. LLM 호출
    response = llm.invoke(messages)

    # 일반적으로 response.content는 문자열이지만,
    # 모델 또는 LangChain 버전에 따라 다른 형식일 가능성에 대비
    if isinstance(response.content, str):
        return response.content

    return str(response.content)


# ============================================================
# 13. 실행 예제
# ============================================================

if __name__ == "__main__":
    question ="우리집 댕댕이가 의자를 파손해 보험금 청구하려 하는데 필요한 서류와 지급 절차는 어떻게 돼?"

    answer = answer_question(
        question=question,
        k=5,
        show_context=True,
    )

    print("\n답변:")
    print(answer)

