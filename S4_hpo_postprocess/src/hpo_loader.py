"""HPO ontology loading utilities."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import obonet  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    obonet = None


@dataclass
class HPOTerm:
    hpo_id: str
    label: str
    synonyms: List[str]
    definition: Optional[str] = None

    @property
    def all_strings(self) -> List[str]:
        base = [self.label]
        base.extend(self.synonyms)
        return [s for s in base if s]


def _normalise_synonyms(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for val in values:
        if not val:
            continue
        clean = val.strip()
        if not clean:
            continue
        if clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def load_hpo_terms(path: Path, fmt: str = "tsv") -> List[HPOTerm]:
    """Load HPO terms from a TSV/JSON/OBO resource.

    The TSV format expects header columns: ``hpo_id``, ``label``, ``synonyms`` (pipe ``|`` separated).
    """
    fmt = fmt.lower()
    if fmt == "tsv":
        return _load_from_tsv(path)
    if fmt == "json":
        return _load_from_json(path)
    if fmt == "obo":
        return _load_from_obo(path)
    raise ValueError(f"Unsupported HPO format: {fmt}")


def _load_from_tsv(path: Path) -> List[HPOTerm]:
    if not path.exists():
        raise FileNotFoundError(f"HPO TSV not found: {path}")
    terms: List[HPOTerm] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            hpo_id = row.get("hpo_id") or row.get("id")
            if not hpo_id:
                continue
            label = row.get("label") or row.get("name") or ""
            synonyms_field = row.get("synonyms") or ""
            synonyms = [s.strip() for s in synonyms_field.split("|") if s.strip()]
            terms.append(HPOTerm(hpo_id=hpo_id.strip(), label=label.strip(), synonyms=_normalise_synonyms(synonyms)))
    return terms


def _load_from_json(path: Path) -> List[HPOTerm]:
    if not path.exists():
        raise FileNotFoundError(f"HPO JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_nodes = []
    if "graphs" in payload:
        for graph in payload["graphs"]:
            raw_nodes.extend(graph.get("nodes", []))
    elif "nodes" in payload:
        raw_nodes.extend(payload.get("nodes", []))

    terms: List[HPOTerm] = []
    for entry in raw_nodes:
        raw_id = entry.get("id") or entry.get("@id")
        if not raw_id:
            continue
        node_id = _convert_id(str(raw_id))
        if not node_id:
            continue
        label = entry.get("lbl") or entry.get("label") or ""
        definition = None
        if "def" in entry:
            definition = entry["def"].get("val") if isinstance(entry["def"], dict) else entry["def"]
        if not definition:
            meta = entry.get("meta", {})
            if isinstance(meta, dict):
                def_val = meta.get("definition")
                if isinstance(def_val, dict):
                    definition = def_val.get("val")
                elif isinstance(def_val, str):
                    definition = def_val

        syns: List[str] = []
        for syn in entry.get("synonym", []):
            if isinstance(syn, dict):
                val = syn.get("val")
                if val:
                    syns.append(val)
            elif isinstance(syn, str):
                syns.append(syn)
        meta = entry.get("meta", {})
        if isinstance(meta, dict):
            for syn in meta.get("synonyms", []):
                if isinstance(syn, dict):
                    val = syn.get("val")
                    if val:
                        syns.append(val)
                elif isinstance(syn, str):
                    syns.append(syn)
        terms.append(HPOTerm(hpo_id=node_id, label=label, synonyms=_normalise_synonyms(syns), definition=definition))
    return terms


def _load_from_obo(path: Path) -> List[HPOTerm]:
    if obonet is None:
        raise ImportError("obonet is not installed; install it or provide TSV/JSON instead")
    if not path.exists():
        raise FileNotFoundError(f"HPO OBO not found: {path}")
    graph = obonet.read_obo(path)
    terms: List[HPOTerm] = []
    for node_id, data in graph.nodes(data=True):
        if not str(node_id).startswith("HP:"):
            continue
        label = data.get("name", "")
        syns = data.get("synonym", [])
        cleaned = []
        for syn in syns:
            # OBO synonym strings look like ""Abnormality" EXACT []"
            if syn.startswith('"'):
                cleaned.append(syn.split('"')[1])
            else:
                cleaned.append(syn)
        terms.append(HPOTerm(hpo_id=node_id, label=label, synonyms=_normalise_synonyms(cleaned), definition=data.get("def")))
    return terms


def build_text_index(terms: List[HPOTerm]) -> Dict[str, List[str]]:
    """Create a mapping from text strings to HPO IDs."""
    index: Dict[str, List[str]] = {}
    for term in terms:
        for text in term.all_strings:
            index.setdefault(text.lower(), []).append(term.hpo_id)
    return index
def _convert_id(raw_id: str) -> Optional[str]:
    if raw_id.startswith("HP:"):
        return raw_id
    if raw_id.startswith("http://purl.obolibrary.org/obo/HP_"):
        tail = raw_id.rsplit("/", 1)[-1]
        return tail.replace("_", ":")
    return None


def load_hpo_canonical_map(path: Path, fmt: str = "json") -> Dict[str, str]:
    """Build mapping from alternative/equivalent IDs to canonical HP IDs.

    For JSON (OBO Graph JSON), alt IDs often appear in meta.basicPropertyValues
    with predicates that include 'hasAlternativeId'.
    """
    alt2prim: Dict[str, str] = {}
    if fmt.lower() != "json" or not path.exists():
        return alt2prim
    import json as _json  # local import
    payload = _json.load(path.open("r", encoding="utf-8"))
    raw_nodes = []
    if "graphs" in payload:
        for graph in payload["graphs"]:
            raw_nodes.extend(graph.get("nodes", []))
    elif "nodes" in payload:
        raw_nodes.extend(payload.get("nodes", []))

    for entry in raw_nodes:
        raw_id = entry.get("id") or entry.get("@id")
        prim = _convert_id(str(raw_id)) if raw_id else None
        if not prim:
            continue
        meta = entry.get("meta", {}) or {}
        bpv = meta.get("basicPropertyValues") or []
        for item in bpv:
            pred = str(item.get("pred", ""))
            val = item.get("val")
            if not val:
                continue
            if "hasAlternativeId" in pred or pred.endswith("#hasAlternativeId"):
                alt = _convert_id(str(val))
                if alt and alt != prim:
                    alt2prim[alt] = prim
        # also ensure identity mapping available for primaries
        alt2prim.setdefault(prim, prim)
    return alt2prim
