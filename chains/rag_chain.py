# ============================================================
# This code is created with the help of AI. As a starting
# point, there was already a working code where a RAG chain
# could be invoked. ChatGPT 5.2 was used to help make this code
# more modular, and to help with integrating images into the RAG.
# All chats can be found at:
# https://chatgpt.com/share/699059a9-13a0-8013-ad55-189696111186
# https://chatgpt.com/share/69905ca0-85c8-8013-bc4f-14ad1864d1db
# https://chatgpt.com/share/69905d3a-cb80-8013-a0be-5b7b799c5954
# ============================================================
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline

from datasets.telemetry import TelemetryLogger
from inputs import TELEMETRY_DB_PATH, RETRIEVER_SEARCH_TYPE, RETRIEVER_SEARCH_KWARGS, COMPRESSOR_SIMILARITY_THRESHOLD, \
    db_info_path
from chains.images_helpers import (
    user_asked_for_images,
    best_image_per_source_semantic,
    build_image_context_from_map,
    extract_cited_source_ids,
    normalize_source_id_citations_to_plain,
)

def infer_llm_meta(llm):
    meta = {"llm_class": llm.__class__.__name__}
    for attr in ["model", "model_name", "model_id", "deployment_name"]:
        if hasattr(llm, attr):
            meta["llm_model"] = getattr(llm, attr)
            break
    for attr in ["temperature", "max_tokens", "top_p"]:
        if hasattr(llm, attr):
            meta[attr] = getattr(llm, attr)
    return meta

def infer_embeddings_meta(embeddings):
    meta = {"embeddings_class": embeddings.__class__.__name__}
    for attr in ["model_name", "model"]:
        if hasattr(embeddings, attr):
            meta["embeddings_model"] = getattr(embeddings, attr)
            break
    return meta

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def format_docs_with_source(docs):
    return "\n\n".join(
        f"Content: {d.page_content}\nSource_id: {d.metadata.get('source','N/A')}"
        for d in docs
    )

def docs_to_telemetry(docs, max_docs=25, preview_chars=300):
    out = []
    for i, d in enumerate(docs[:max_docs]):
        md = dict(d.metadata or {})
        md_small = {}
        for k, v in md.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                md_small[k] = v
        out.append({
            "i": i,
            "source": md.get("source", "N/A"),
            "image_key": md.get("image_key"),
            "metadata": md_small,
            "preview": (d.page_content or "")[:preview_chars]
        })
    return out

def load_evdb_slim_text(evdb_slim_path: str, max_cars: int | None = None) -> str:
    """
    Loads evdb_slim JSON and returns a compact, prompt-friendly listing.
    Keep it short: the model only needs to know which cars exist.
    """
    p = Path(evdb_slim_path)
    data = json.loads(p.read_text(encoding="utf-8"))

    # Expect list[dict] with at least: name, brand, url (your slimmer format)
    # Create a readable index line per car.
    lines: list[str] = []
    for item in data:
        name = item.get("name") or "Unknown"
        brand = item.get("brand") or "Unknown"
        url = item.get("url") or ""
        lines.append(f"- {brand} | {name} | {url}")

    if max_cars is not None:
        lines = lines[:max_cars]

    return "EVDB_CAR_INDEX:\n" + "\n".join(lines)

def merge_context(dynamic_context: str, database_info_context: str) -> str:
    """
    Ensures the EVDB index is always present, without exploding token usage too much.
    """
    if not database_info_context:
        return dynamic_context
    return f"{dynamic_context}\n\n---\n\n{database_info_context}"

def build_rag_chain(
    llm,
    embeddings,
    chroma_path,
    prompt_rephrase,
    prompt_rag,
    trimmer,
    collection_name,
    chroma_path2=None,
):
    # Vector DB 1
    chroma_db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    retriever = chroma_db.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs=RETRIEVER_SEARCH_KWARGS,
    )

    # Optional DB 2
    if chroma_path2 is not None:
        chroma_db2 = Chroma(
            persist_directory=chroma_path2,
            embedding_function=embeddings,
            collection_name="polestar_website",
        )
        retriever2 = chroma_db2.as_retriever(
            search_type=RETRIEVER_SEARCH_TYPE,
            search_kwargs=RETRIEVER_SEARCH_KWARGS,
        )
        retriever = EnsembleRetriever(retrievers=[retriever, retriever2], weights=[0.5, 0.5])

    compressor = DocumentCompressorPipeline(
        transformers=[EmbeddingsFilter(embeddings=embeddings, similarity_threshold=COMPRESSOR_SIMILARITY_THRESHOLD)]
    )

    retriever_vdb = ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=compressor,
    )

    retriever_hist = create_history_aware_retriever(
        llm=llm,
        retriever=retriever_vdb,
        prompt=prompt_rephrase,
    )

    telemetry = TelemetryLogger(TELEMETRY_DB_PATH)

    static_model_meta = {
        **infer_llm_meta(llm),
        **infer_embeddings_meta(embeddings),
        "retriever": {
            "search_type": RETRIEVER_SEARCH_TYPE,
            "search_kwargs": dict(RETRIEVER_SEARCH_KWARGS),
            "compressor_similarity_threshold": COMPRESSOR_SIMILARITY_THRESHOLD,
            "collection_name": collection_name,
            "has_second_db": chroma_path2 is not None,
        },
    }

    def step_all(x: dict) -> dict:
        """
        Input requires:
          question, client_context, chat_history, session_id, sliders(optional), prompt_templates(optional)
        Output:
          answer, images_abs, images_meta, run_id
        """
        run_id = uuid.uuid4().hex
        session_id = x.get("session_id") or "default_session"

        trimmed_hist = trimmer.invoke(x.get("chat_history", []))

        # explicit rephrase for telemetry
        rephrase_msgs = prompt_rephrase.format_messages(
            chat_history=trimmed_hist,
            input=x["question"],
        )
        user_question_rephrased = StrOutputParser().invoke(llm.invoke(rephrase_msgs)).strip()

        docs = retriever_hist.invoke({"input": user_question_rephrased, "chat_history": trimmed_hist})

        retrieved_source_ids = []
        seen = set()
        for d in docs:
            src = (d.metadata or {}).get("source")
            if src and src not in seen:
                seen.add(src)
                retrieved_source_ids.append(src)

        context_dynamic = format_docs_with_source(docs)
        db_info_text = load_evdb_slim_text(db_info_path)
        context = merge_context(context_dynamic, db_info_text)

        # images: pick best per Source_id if user asked
        want_images = user_asked_for_images(x["question"])
        image_map = best_image_per_source_semantic(docs, x["question"], max_sources=10) if want_images else {}
        image_context = build_image_context_from_map(image_map)

        # answer
        msgs = prompt_rag.format_messages(
            context=context,
            image_context=image_context,
            client_context=x.get("client_context", ""),
            question=x["question"],
            chat_history=trimmed_hist,
        )
        prompt_rendered = [{"role": getattr(m, "type", "unknown"), "content": m.content} for m in msgs]
        answer_raw = StrOutputParser().invoke(llm.invoke(msgs))

        # normalize citations so we can reliably match Source_id
        answer = normalize_source_id_citations_to_plain(answer_raw)

        # if user did NOT ask images, strip any markdown image tags the model invented
        if not want_images:
            # remove lines like ![...](...)
            answer = "\n".join([ln for ln in answer.splitlines() if not ln.strip().startswith("![")]).strip()

        cited_sources = extract_cited_source_ids(answer)
        if cited_sources:
            image_map = {src: image_map[src] for src in cited_sources if src in image_map}
        else:
            image_map = {}

        images_abs = [im["abs_path"] for im in image_map.values()]
        images_meta = list(image_map.values())

        telemetry.log_turn({
            "run_id": run_id,
            "session_id": session_id,
            "ts_utc": utc_now_iso(),
            "user_question_raw": x.get("question"),
            "user_question_rephrased": user_question_rephrased,
            "answer_text": answer,
            "prompt_templates": x.get("prompt_templates", {}),
            "prompt_rendered": prompt_rendered,
            "sliders": x.get("sliders", {}),
            "retrieved_docs": docs_to_telemetry(docs),

            "retrieved_source_ids": retrieved_source_ids,  # <-- NEW

            "images": [
                {
                    "source_id": im.get("source_id"),
                    "alt": im.get("alt", ""),
                    "sha256": im.get("sha256"),
                    "rel_path": im.get("rel_path"),
                    "url": im.get("url", ""),
                    "abs_path": im.get("abs_path"),
                } for im in images_meta
            ],

            "model": {**static_model_meta, **(x.get("model", {}) or {})},  # <-- CHANGED
            "metrics": x.get("metrics", {}),
        })

        return {
            "answer": answer,
            "images_abs": images_abs,
            "images_meta": images_meta,
            "run_id": run_id,
            "user_question_rephrased": user_question_rephrased,
            "docs": docs,  # optional, but handy
        }

    chain = RunnableLambda(step_all)
    return chain, retriever_hist
