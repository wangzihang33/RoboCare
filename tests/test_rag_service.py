from rag.rag_service import build_local_rag_retriever


def test_local_rag_service_uses_rrf_candidates_with_second_stage_reranking():
    calls = []

    class FakeVectorService:
        def get_fusion_rerank_retriever(self, *, k):
            calls.append(k)
            return "reranked-retriever"

    retriever = build_local_rag_retriever(FakeVectorService(), top_k=3)

    assert retriever == "reranked-retriever"
    assert calls == [3]


def test_evaluation_rag_mode_uses_rrf_without_second_stage_reranking():
    calls = []

    class FakeVectorService:
        def get_fusion_retriever(self, *, k):
            calls.append(("rrf", k))
            return "rrf-retriever"

        def get_fusion_rerank_retriever(self, *, k):
            calls.append(("rerank", k))
            return "reranked-retriever"

    retriever = build_local_rag_retriever(
        FakeVectorService(),
        top_k=3,
        use_reranker=False,
    )

    assert retriever == "rrf-retriever"
    assert calls == [("rrf", 3)]
