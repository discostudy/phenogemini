#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import pandas as pd


def sent_split(text: str):
    return re.split(r'(?<=[.!?。！？])\s+', text)


def main():
    ap = argparse.ArgumentParser(description='Build multi-view OMIM queries: gene/def/case views per OMIM.')
    ap.add_argument('--omim_json', required=True, help='Path to OMIM entries JSON')
    ap.add_argument('--sections', default='molecularGenetics,genotypePhenotypeCorrelations', help='Comma list of OMIM sections to use')
    ap.add_argument('--case_cues', default='patient,proband,case,heterozyg,homozyg,mutation,variant,c\.,p\.,exon,deletion,insertion', help='Regex keywords for case sentences (comma separated -> joined by |)')
    ap.add_argument('--def_sentences', type=int, default=2, help='Number of leading sentences for def view')
    ap.add_argument('--case_max_sentences', type=int, default=3, help='Max case sentences to include')
    ap.add_argument('--out', default='data/queries_omim_multi.tsv', help='Output TSV with columns: query_id, query_text')
    args = ap.parse_args()

    with open(args.omim_json, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    sections = {s.strip() for s in args.sections.split(',') if s.strip()}
    cues = re.compile('(' + '|'.join([re.escape(x.strip()) for x in args.case_cues.split(',') if x.strip()]) + ')', re.I)

    rows = []
    for e in entries:
        try:
            omim = f"OMIM:{e.get('mimNumber')}"
        except Exception:
            continue

        # gene view: approved symbol + aliases
        gm = (e.get('geneMap') or {})
        appr = gm.get('approvedGeneSymbols')
        aliases = []
        if isinstance(gm.get('geneSymbols'), str):
            aliases = [x.strip() for x in gm['geneSymbols'].split(',') if x.strip()]
        gene_view = '; '.join([x for x in [appr] + aliases if x])
        if gene_view:
            rows.append((f"{omim}#gene", gene_view))

        # collect section texts
        texts = []
        for sec in e.get('textSectionList', []) or []:
            s = sec.get('textSection') or {}
            name = s.get('textSectionName')
            content = s.get('textSectionContent')
            if name in sections and isinstance(content, str):
                texts.append(content)
        if not texts:
            continue
        text = ' '.join(texts)
        ss = [s for s in sent_split(text) if s and isinstance(s, str)]
        if not ss:
            continue

        # def view: first N sentences
        def_view = ' '.join(ss[: max(1, args.def_sentences)])
        rows.append((f"{omim}#def", def_view))

        # case view: sentences with cues
        case_sents = [s for s in ss if cues.search(s)]
        if case_sents:
            rows.append((f"{omim}#case", ' '.join(case_sents[: max(1, args.case_max_sentences)])))

    df = pd.DataFrame(rows, columns=['query_id', 'query_text']).drop_duplicates()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_csv(args.out, sep='\t', index=False)
    print(f"[OK] wrote {args.out} views={df.shape[0]} queries={df['query_id'].nunique()}")


if __name__ == '__main__':
    main()

