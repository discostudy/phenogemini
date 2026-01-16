#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from ek_phenopub.source.pubmed_repository import PubRepository

PM_PAT = re.compile(r'^PM:(\d+)$')


def _pm_to_int(doc_id: str) -> int:
    m = PM_PAT.match(doc_id)
    if not m:
        raise ValueError(f'Invalid doc_id (expect PM:<pmid>): {doc_id}')
    return int(m.group(1))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def load_queries(queries_tsv: str) -> Dict[str, str]:
    df = pd.read_csv(queries_tsv, sep='\t')
    return dict(zip(df['query_id'].astype(str), df['query_text'].astype(str)))


def load_qdoc_pairs(qdoc_pairs_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(qdoc_pairs_tsv, sep='\t')
    # 只关心 query_id / doc_id
    return df[['query_id', 'doc_id']].drop_duplicates().reset_index(drop=True)


def load_docs_list(docs_tsv: str) -> list:
    # 支持两列: doc_id,label 或单列: doc_id
    df = pd.read_csv(docs_tsv, sep='\t')
    col = 'doc_id' if 'doc_id' in df.columns else df.columns[0]
    return [str(x) for x in df[col].tolist()]


def fetch_docs_text(repo: PubRepository, doc_ids: List[str]) -> Dict[str, str]:
    out = {}
    for d in tqdm(doc_ids, desc='fetch docs'):
        pmid = _pm_to_int(d)
        full = repo.get_pub_full(f'PM:{pmid}')
        if full and 'id' in full:
            rec = repo.get_pub_jtak_by_id(full['id'])
        else:
            rec = repo.get_pub_jtak_by_id(pmid)
        if not rec or not rec.get('text'):
            continue
        out[d] = rec['text']
    return out


def safe_name(model_name: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9_.-]+', '_', model_name)
    # Avoid hidden filenames like '.model' when model_name starts with './'
    s = s.lstrip('._')
    return s or 'model'


def run_model(model_name: str,
              queries: Dict[str, str],
              repo_path: str,
              out_path: str,
              batch_size: int = 128,
              qdoc_df: pd.DataFrame = None,
              docs_list: list = None,
              topk_out: int = 2000,
              devices: List[str] = None):
    # 文档候选来源：优先 docs_list（全局池），否则使用 qdoc_df（每查询不同候选）
    if docs_list is not None:
        dids = sorted(set(docs_list))
        # qids 取所有查询
        qids = sorted(queries.keys())
    else:
        qids = sorted(set(qdoc_df['query_id'].astype(str)))
        dids = sorted(set(qdoc_df['doc_id'].astype(str)))

    # 取文本
    repo = PubRepository(repo_path)
    doc_texts_map = fetch_docs_text(repo, dids)
    dids = [d for d in dids if d in doc_texts_map]

    # Debug dump: queries and docs (pmid + jtak) used for this run
    try:
        base_dir = os.path.dirname(os.path.dirname(out_path))
        run_name = safe_name(model_name)
        debug_dir = os.path.join(base_dir, 'results', 'debug', run_name)
        os.makedirs(debug_dir, exist_ok=True)
        # Queries used
        try:
            import pandas as _pd
            q_used = [q for q in qids if q in queries]
            _pd.DataFrame({'query_id': q_used, 'query_text': [queries[q] for q in q_used]}).to_csv(os.path.join(debug_dir, 'queries_used.tsv'), sep='\t', index=False)
        except Exception:
            pass
        # Docs used
        try:
            import pandas as _pd
            _pd.DataFrame({'doc_id': dids, 'pmid': [str(_pm_to_int(d)) for d in dids], 'jtak': [doc_texts_map[d] for d in dids]}).to_csv(os.path.join(debug_dir, 'docs_used.tsv'), sep='\t', index=False)
        except Exception:
            pass
    except Exception:
        pass

    # 若使用 qdoc_df，过滤掉无文本文档的 pair
    if qdoc_df is not None:
        qdoc_df = qdoc_df[qdoc_df['doc_id'].isin(dids)].reset_index(drop=True)

    # 组装向量
    devices = [dv.strip() for dv in (devices or []) if dv.strip()]
    model = SentenceTransformer(model_name)
    pool = None
    if devices:
        print(f'[info] SBERT multi-device: {devices}')
        pool = model.start_multi_process_pool(target_devices=devices)

    try:
        # Queries
        qids = [q for q in qids if q in queries]
        q_texts = [queries[q] for q in qids]
        if pool is not None:
            q_emb = model.encode_multi_process(q_texts, pool, batch_size=batch_size, normalize_embeddings=False)
        else:
            q_emb = model.encode(q_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
        q_emb = _l2_normalize(np.asarray(q_emb, dtype=np.float32))
        q_index = {q: i for i, q in enumerate(qids)}

        # Docs
        d_texts = [doc_texts_map[d] for d in dids]
        if pool is not None:
            d_emb = model.encode_multi_process(d_texts, pool, batch_size=batch_size, normalize_embeddings=False)
        else:
            d_emb = model.encode(d_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
        d_emb = _l2_normalize(np.asarray(d_emb, dtype=np.float32))
        d_index = {d: i for i, d in enumerate(dids)}
    finally:
        if pool is not None:
            model.stop_multi_process_pool(pool)

    rows: List[Tuple[str, str, float]] = []
    if docs_list is not None:
        # 全局文档池：对每个查询在全局池上打分，输出 topk_out
        d_mat = d_emb.T  # (d, n)
        for q in tqdm(qids, desc='score (global pool)'):
            qi = q_index[q]
            qv = q_emb[qi:qi+1]  # (1,d)
            scores = (qv @ d_mat).reshape(-1)
            if topk_out and topk_out > 0 and topk_out < len(dids):
                import numpy as _np
                top_idx = _np.argpartition(-scores, topk_out)[:topk_out]
                # 排序
                top_idx = top_idx[_np.argsort(-scores[top_idx])]
            else:
                import numpy as _np
                top_idx = _np.argsort(-scores)
            for idx in top_idx:
                rows.append((q, dids[idx], float(scores[idx])))
    else:
        # 每查询不同候选：仅在该查询候选上打分
        import numpy as _np
        for q, group in tqdm(qdoc_df.groupby('query_id'), total=len(qdoc_df.groupby('query_id'))):
            if q not in q_index:
                continue
            qi = q_index[q]
            qv = q_emb[qi:qi+1]
            cand = [d for d in group['doc_id'].tolist() if d in d_index]
            if not cand:
                continue
            idxs = [d_index[d] for d in cand]
            dv = d_emb[idxs]
            scores = (qv @ dv.T).reshape(-1)
            order = _np.argsort(-scores)
            if topk_out and topk_out > 0:
                order = order[:topk_out]
            for d, s in zip([cand[i] for i in order], scores[order].tolist()):
                rows.append((q, d, float(s)))

    out_df = pd.DataFrame(rows, columns=['query_id', 'doc_id', 'score'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f'[OK] wrote {out_path} with {len(out_df)} rows')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', required=True)
    ap.add_argument('--qdoc_pairs', default=None, help='可选：每查询候选对 (query_id, doc_id)')
    ap.add_argument('--docs', default=None, help='可选：全局文档池 (doc_id[\tlabel])')
    ap.add_argument('--repo', required=True)
    ap.add_argument('--models', required=True, help='逗号分隔的模型列表')
    ap.add_argument('--out_dir', default='runs')
    ap.add_argument('--out_file', default=None, help='如果提供，则直接写到该文件（仅当 models 只有一个时使用）')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--topk_out', type=int, default=2000, help='每个查询输出的最多文档数')
    ap.add_argument('--devices', default='', help='逗号分隔的设备列表 (如 cuda:0,cuda:1)')
    args = ap.parse_args()

    queries = load_queries(args.queries)
    qdoc_df = load_qdoc_pairs(args.qdoc_pairs) if args.qdoc_pairs else None
    docs_list = load_docs_list(args.docs) if args.docs else None

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    device_list = [d.strip() for d in args.devices.split(',') if d.strip()]
    for m in models:
        if args.out_file and len(models) == 1:
            out_path = args.out_file
        else:
            out_path = os.path.join(args.out_dir, f'{safe_name(m)}.tsv')
        print(f'=== Running SBERT model: {m} -> {out_path}')
        run_model(m, queries, args.repo, out_path, batch_size=args.batch_size, qdoc_df=qdoc_df, docs_list=docs_list, topk_out=args.topk_out, devices=device_list)

if __name__ == '__main__':
    main()

