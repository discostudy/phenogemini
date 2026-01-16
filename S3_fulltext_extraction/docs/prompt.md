Extract and catalog relevant information for each patient mentioned in this medical literature.

This is very important to my career. Please ensure that no patient is left out and each patient's data is presented individually.

Please extract the following information:
- Phenotypes: Extract phenotypes/clinical features EXACTLY as described in the original text (no HPO codes, just the raw descriptions from the document)
- Causative_Gene: Identify the gene responsible using the gene symbol
- Causative_Variant: Extract the causative variant using HGVS nomenclature
- Disease: Extract the disease name
- Testing method: What genetic testing was performed (WGS, WES, Panel, CMA, etc.)
- Reference genome and chromosomal coordinates if available

Present your findings in a structured JSON format:

{
  "document_info": {
    "file_name": "{file_name}",
    "total_patients_found": <number>,
    "extraction_confidence": "high/medium/low",
    "explanation": "brief explanation of what was found"
  },
  "patients": [
    {
      "patient_id": "unique_identifier",
      "demographics": {
        "age": "age_information",
        "sex": "male/female/unknown",
        "ethnicity": "ethnicity_if_available",
        "nationality": "nationality_if_available",
        "occupation": "occupation_if_available"
      },
      "phenotypes": [
        "exact phenotype description from the document",
        "another phenotype as written in the paper"
      ],
      "genotype": {
        "gene": "gene_symbol",
        "testing_method": "WGS/WES/Panel/CMA/Sanger/other (specify if other)",
        "variants": [
          {
            "transcript": "transcript_id_if_available (e.g., NM_000123.4)",
            "hgvs_cdna": "cDNA_variant",
            "hgvs_protein": "protein_variant",
            "reference_genome": "GRCh37/GRCh38/unknown",
            "chromosomal_position": "chr:position if provided (e.g., chr1:12345678)",
            "zygosity": "heterozygous/homozygous/hemizygous/unknown",
            "compound_heterozygous": "yes/no",
            "compound_heterozygous_phase": "in_trans/in_cis/unknown/not_applicable",
            "inheritance_pattern": "paternal/maternal/de_novo/unknown",
            "pathogenicity": "pathogenic/likely_pathogenic/variant_of_uncertain_significance/benign/likely_benign/unknown"
          }
        ]
      },
      "diagnosis": "disease_name"
    }
  ]
}

IMPORTANT INSTRUCTIONS:
1. For phenotypes: Extract the EXACT text from the document - do not paraphrase or categorize
2. Phenotypes should be a simple array of strings, each containing the raw description from the paper
3. Do NOT add HPO codes, severity scores, or onset information unless explicitly stated in the original text
4. If multiple patients share similar features, still list each patient separately
5. Keep the extraction focused and simple - quality over quantity
6. For zygosity: Use ONLY "heterozygous", "homozygous", "hemizygous", or "unknown". Do NOT use "compound_heterozygous" as a zygosity value
7. For compound_heterozygous: Always include this field. Use "yes" if the patient has two different heterozygous variants in the same gene (compound heterozygous), otherwise "no"
8. For compound_heterozygous_phase: Always include this field. If compound_heterozygous is "yes", specify "in_trans" (variants on opposite chromosomes), "in_cis" (variants on same chromosome), or "unknown". If compound_heterozygous is "no", use "not_applicable"
9. For inheritance_pattern: Use ONLY "paternal", "maternal", "de_novo", or "unknown"
10. For testing_method: Extract the specific genetic test used (WGS=Whole Genome Sequencing, WES=Whole Exome Sequencing, Panel=Gene Panel, CMA=Chromosomal Microarray, etc.)
11. For pathogenicity: Use "pathogenic", "likely_pathogenic", "variant_of_uncertain_significance", "benign", "likely_benign", or "unknown"
12. For reference_genome and chromosomal_position: Extract these ONLY if explicitly mentioned in the paper

