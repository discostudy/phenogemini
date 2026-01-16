#!/usr/bin/env python3
"""
Hallucination Detection Module
Validates extraction results against original markdown content
"""
import os
import re
import csv
import logging
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import pandas as pd
from config import config

logger = logging.getLogger(__name__)

class HallucinationChecker:
    """Check for hallucinations in extracted medical data"""
    
    def __init__(self, markdown_tsv_path: str = None):
        """Initialize hallucination checker"""
        if markdown_tsv_path is None:
            # Use path from config (reads from .env or defaults to /data/markdown_content.tsv)
            markdown_tsv_path = config.markdown_tsv_path
        
        self.markdown_tsv_path = Path(markdown_tsv_path)
        self.content_df = None  # Will load immediately to avoid multiple reads in parallel tasks
        logger.info(f"Initialized HallucinationChecker with TSV: {self.markdown_tsv_path}")
        
        # Load TSV immediately to avoid multiple reads in parallel processing
        self._load_content_df()
    
    def _load_content_df(self):
        """Load the markdown content TSV file using pandas"""
        if self.content_df is not None:
            return
        
        if not self.markdown_tsv_path.exists():
            logger.warning(f"Markdown TSV file not found: {self.markdown_tsv_path}")
            return
        
        try:
            logger.info(f"Loading markdown content from {self.markdown_tsv_path}")
            self.content_df = pd.read_csv(str(self.markdown_tsv_path), sep="\t")
            logger.info(f"Loaded {len(self.content_df)} records from markdown_content.tsv")
        except Exception as e:
            logger.error(f"Error loading markdown content TSV: {e}")
            self.content_df = pd.DataFrame()
    
    def get_pmid_content(self, pmid: str) -> Optional[str]:
        """Get markdown content for a specific PMID"""
        # TSV is already loaded in __init__, no need to load again
        
        if self.content_df is None or self.content_df.empty:
            logger.warning("No content dataframe available")
            return None
        
        try:
            # Find the row with matching PMID
            pmid_rows = self.content_df[self.content_df.iloc[:, 0].astype(str) == str(pmid)]
            
            if pmid_rows.empty:
                logger.debug(f"No content found for PMID {pmid}")
                return None
            
            # Get the content from the third column (index 2)
            if len(pmid_rows.columns) < 3:
                logger.warning(f"TSV file has less than 3 columns")
                return None
            
            content = pmid_rows.iloc[0, 2]
            if pd.isna(content):
                logger.debug(f"Content is NaN for PMID {pmid}")
                return None
            
            content = str(content).strip()
            logger.debug(f"Found content for PMID {pmid}: {len(content)} chars")
            return content
            
        except Exception as e:
            logger.error(f"Error reading content for PMID {pmid}: {e}")
            return None
    
    def extract_gene_symbols(self, text: str) -> Set[str]:
        """Extract gene symbols from text"""
        if not text:
            return set()
        
        # Common gene symbol patterns
        gene_patterns = [
            r'\b[A-Z][A-Z0-9]{1,9}\b',  # Standard gene symbols (2-10 chars, starts with letter)
            r'\b[A-Z][A-Za-z0-9-]*\d[A-Za-z0-9-]*\b',  # Genes with numbers
        ]
        
        genes = set()
        for pattern in gene_patterns:
            matches = re.findall(pattern, text)
            genes.update(matches)
        
        # Filter out common false positives
        false_positives = {
            'DNA', 'RNA', 'PCR', 'HIV', 'AIDS', 'USA', 'UK', 'EU', 'WHO', 'FDA',
            'MRI', 'CT', 'EEG', 'ECG', 'IV', 'IM', 'PO', 'BID', 'TID', 'QID',
            'AM', 'PM', 'ER', 'ICU', 'OR', 'ED', 'MD', 'PhD', 'BSc', 'MSc'
        }
        
        return genes - false_positives
    
    def extract_variants(self, text: str) -> Set[str]:
        """Extract HGVS variants from text"""
        if not text:
            return set()
        
        variant_patterns = [
            r'c\.\d+[ACGT]>[ACGT]',  # cDNA substitutions
            r'c\.\d+_\d+del[ACGT]*',  # cDNA deletions
            r'c\.\d+_\d+ins[ACGT]+',  # cDNA insertions
            r'p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}',  # Protein substitutions
            r'p\.[A-Z][a-z]{2}\d+\*',  # Nonsense mutations
            r'p\.[A-Z][a-z]{2}\d+fs',  # Frameshift
            r'p\.[A-Z][a-z]{2}\d+del',  # Protein deletions
        ]
        
        variants = set()
        for pattern in variant_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            variants.update(matches)
        
        return variants
    
    def extract_phenotypes(self, text: str) -> Set[str]:
        """Extract phenotype terms from text"""
        if not text:
            return set()
        
        # Medical terminology patterns
        phenotype_patterns = [
            r'\b\w*trophy\b',  # atrophy, dystrophy, etc.
            r'\b\w*pathy\b',   # neuropathy, myopathy, etc.
            r'\b\w*plasia\b',  # dysplasia, hyperplasia, etc.
            r'\b\w*osis\b',    # fibrosis, necrosis, etc.
            r'\b\w*emia\b',    # anemia, leukemia, etc.
            r'\b\w*uria\b',    # proteinuria, hematuria, etc.
        ]
        
        phenotypes = set()
        text_lower = text.lower()
        
        for pattern in phenotype_patterns:
            matches = re.findall(pattern, text_lower)
            phenotypes.update(matches)
        
        return phenotypes
    
    def calculate_overlap_score(self, extracted_set: Set[str], reference_set: Set[str]) -> float:
        """Calculate overlap score between two sets"""
        if not extracted_set and not reference_set:
            return 1.0
        if not extracted_set or not reference_set:
            return 0.0
        
        intersection = len(extracted_set & reference_set)
        union = len(extracted_set | reference_set)
        
        return intersection / union if union > 0 else 0.0
    
    def check_extraction(self, pmid: str, extraction_result: Dict) -> Dict:
        """Check extraction result against original content for hallucinations"""
        
        # Get original content
        original_content = self.get_pmid_content(pmid)
        if not original_content:
            return {
                'total_variants': 0,
                'verified_variants': 0,
                'suspicious_variants': 0,
                'credibility_score': 0.5,  # Neutral when no reference
                'extraction_source': 'gpt41_responses_api',
                'verification_method': 'markdown_content_comparison',
                'reference_available': False,
                'verification_details': {
                    'error': 'No reference content found for verification'
                }
            }
        
        # Extract information from original content
        original_genes = self.extract_gene_symbols(original_content)
        original_variants = self.extract_variants(original_content)
        original_phenotypes = self.extract_phenotypes(original_content)
        
        # Extract information from GPT results
        extracted_genes = set()
        extracted_variants = set()
        extracted_phenotypes = set()
        total_variants = 0
        
        patients = extraction_result.get('patients', [])
        for patient in patients:
            # Extract phenotypes
            phenotypes = patient.get('phenotypes', [])
            for phenotype in phenotypes:
                if isinstance(phenotype, str):
                    extracted_phenotypes.update(self.extract_phenotypes(phenotype))
            
            # Extract genetic information
            genotype = patient.get('genotype', {})
            gene = genotype.get('gene', '')
            if gene:
                extracted_genes.add(gene)
            
            variants = genotype.get('variants', [])
            for variant in variants:
                total_variants += 1
                hgvs_cdna = variant.get('hgvs_cdna', '')
                hgvs_protein = variant.get('hgvs_protein', '')
                
                if hgvs_cdna:
                    extracted_variants.update(self.extract_variants(hgvs_cdna))
                if hgvs_protein:
                    extracted_variants.update(self.extract_variants(hgvs_protein))
        
        # Calculate overlap scores
        gene_overlap = self.calculate_overlap_score(extracted_genes, original_genes)
        variant_overlap = self.calculate_overlap_score(extracted_variants, original_variants)
        phenotype_overlap = self.calculate_overlap_score(extracted_phenotypes, original_phenotypes)
        
        # Calculate overall credibility score
        scores = []
        if original_genes or extracted_genes:
            scores.append(gene_overlap)
        if original_variants or extracted_variants:
            scores.append(variant_overlap)
        if original_phenotypes or extracted_phenotypes:
            scores.append(phenotype_overlap)
        
        credibility_score = sum(scores) / len(scores) if scores else 0.5
        
        # Determine suspicious variants
        verified_variants = len(extracted_variants & original_variants)
        suspicious_variants = max(0, total_variants - verified_variants)
        
        verification_details = {
            # 原始内容统计
            'original_content_length': len(original_content),  # 原始markdown内容字符数
            
            # 重叠度评分 (0-1, 越接近1越可信)
            'gene_overlap_score': gene_overlap,      # 基因重叠度
            'variant_overlap_score': variant_overlap,  # 变异重叠度  
            'phenotype_overlap_score': phenotype_overlap,  # 表型重叠度
            
            # 提取结果对比
            'extracted_genes': list(extracted_genes),     # GPT提取的基因
            'reference_genes': list(original_genes),      # 原文中的基因
            'extracted_variants': list(extracted_variants),  # GPT提取的变异
            'reference_variants': list(original_variants),   # 原文中的变异
            
            # 验证结果分析
            'common_genes': list(extracted_genes & original_genes),      # 共同基因(正确提取)
            'common_variants': list(extracted_variants & original_variants),  # 共同变异(正确提取)
            'novel_genes': list(extracted_genes - original_genes),       # 新发现基因(可能幻觉)
            'novel_variants': list(extracted_variants - original_variants)   # 新发现变异(可能幻觉)
        }
        
        logger.info(f"PMID {pmid} hallucination check - Credibility: {credibility_score:.3f}, "
                   f"Variants: {verified_variants}/{total_variants} verified")
        
        return {
            'total_variants': total_variants,
            'verified_variants': verified_variants,
            'suspicious_variants': suspicious_variants,
            'credibility_score': credibility_score,
            'extraction_source': 'gpt41_responses_api',
            'verification_method': 'markdown_content_comparison',
            'reference_available': True,
            'verification_details': verification_details
        }

def test_hallucination_checker():
    """Test function for the hallucination checker"""
    checker = HallucinationChecker()
    
    # Test with a sample PMID
    test_pmid = "1000869"
    content = checker.get_pmid_content(test_pmid)
    
    if content:
        print(f"Found content for PMID {test_pmid}: {len(content)} characters")
        print("First 200 characters:", content[:200])
        
        # Test extraction methods
        genes = checker.extract_gene_symbols(content)
        variants = checker.extract_variants(content)
        phenotypes = checker.extract_phenotypes(content)
        
        print(f"Extracted genes: {genes}")
        print(f"Extracted variants: {variants}")
        print(f"Extracted phenotypes: {phenotypes}")
    else:
        print(f"No content found for PMID {test_pmid}")

if __name__ == "__main__":
    test_hallucination_checker()