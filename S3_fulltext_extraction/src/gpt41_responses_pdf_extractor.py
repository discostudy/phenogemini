#!/usr/bin/env python3
"""
GPT-4.1 + Responses API + File Search based medical literature PDF information extractor
Using the new Responses API to replace the deprecated Assistants API
"""
import os
import sys
import json
import time
import logging
import threading
import pandas as pd
import openai
import httpx
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from validator import validate_extraction_result, get_quality_score
from hallucination_checker import HallucinationChecker

# Setup logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure logging to both file and console
log_file = os.path.join(log_dir, f'extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GPT41ResponsesPDFExtractor:
    """GPT-4.1 Responses API based PDF extractor"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize extractor"""
        # Setup proxy if enabled
        if config.proxy_enabled:
            os.environ['HTTP_PROXY'] = config.http_proxy
            os.environ['HTTPS_PROXY'] = config.https_proxy
        
        if api_key:
            config.override_api_key(api_key)
        
        self.api_key = config.openai_api_key
        self.api_config = config.get_openai_config()
        self.output_dir = config.extraction_output_dir
        self.checkpoint_file = config.checkpoint_file
        self.project_root = config.project_root
        
        # Initialize OpenAI client once and reuse
        timeout = httpx.Timeout(
            connect=30.0,  # Connection timeout 30 seconds
            read=300.0,    # Read timeout 5 minutes
            write=30.0,    # Write timeout 30 seconds
            pool=30.0      # Connection pool timeout 30 seconds
        )
        
        # Create a shared client with connection pooling
        self.client = openai.OpenAI(
            api_key=self.api_key,
            timeout=timeout,
            max_retries=3,
            # Limit connection pool size to prevent resource exhaustion
            http_client=httpx.Client(
                timeout=timeout,
                limits=httpx.Limits(
                    max_connections=10,  # Total connections
                    max_keepalive_connections=5  # Keepalive connections
                )
            )
        )
        
        # Create necessary directories
        config.ensure_directories()
        
        # Ensure logs directory exists
        log_dir = os.path.join(self.project_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize hallucination checker
        self.hallucination_checker = HallucinationChecker()
        
        # Thread lock for checkpoint operations
        self.checkpoint_lock = threading.Lock()
        
        logger.info(f"Initializing GPT41ResponsesPDFExtractor with model: {config.openai_model}")
        logger.info(f"PDF directory: {config.pdf_directory}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Hallucination checker initialized")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if hasattr(self, 'client') and hasattr(self.client, '_client'):
                # Close the underlying HTTP client
                self.client._client.close()
                logger.debug("Closed OpenAI client connection pool")
        except Exception as e:
            logger.warning(f"Error closing client: {e}")
    
    def get_pdf_files(self, directory: str) -> List[str]:
        """Get all PDF files in directory"""
        pdf_files = []
        
        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return pdf_files
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf') and not file.startswith('.'):
                    pdf_path = os.path.join(root, file)
                    pdf_files.append(pdf_path)
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        return sorted(pdf_files)
    
    def validate_pdf_file(self, pdf_path: str) -> tuple[bool, str]:
        """Validate PDF file"""
        try:
            if not os.path.exists(pdf_path):
                return False, "File does not exist"
            
            file_size = os.path.getsize(pdf_path)
            if file_size == 0:
                return False, "File is empty"
            
            # Responses API has PDF size limit, generally recommended less than 32MB
            if file_size > 32 * 1024 * 1024:  # 32MB
                return False, f"File too large ({file_size / 1024 / 1024:.1f}MB), should be less than 32MB"
            
            return True, f"File valid ({file_size / 1024:.1f}KB)"
            
        except Exception as e:
            return False, f"File validation error: {e}"
    
    def load_checkpoint(self, checkpoint_file: str) -> List[Dict]:
        """Load checkpoint file"""
        if not os.path.exists(checkpoint_file):
            logger.info(f"Checkpoint file does not exist: {checkpoint_file}")
            return []
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded checkpoint: {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return []
    
    def save_checkpoint(self, results: List[Dict], checkpoint_file: str):
        """Save checkpoint file (thread-safe)"""
        with self.checkpoint_lock:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    checkpoint_dir = os.path.dirname(checkpoint_file)
                    if checkpoint_dir:
                        os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    logger.debug(f"Saved checkpoint: {len(results)} results")
                    break  # Success, exit retry loop
                except (OSError, IOError) as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to save checkpoint after {max_retries} attempts: {e}")
                        # Don't raise, just log the error
                        break
                    else:
                        logger.warning(f"Checkpoint save attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(0.1)  # Short delay before retry
    
    def save_individual_result(self, result: Dict, pdf_path: str):
        """Save extraction result to output directory"""
        try:
            file_name = os.path.basename(pdf_path)
            # Use PMID as filename
            pmid = file_name.replace('.pdf', '')
            result_file = os.path.join(self.output_dir, f"{pmid}.json")
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved extraction result: {result_file}")
            
        except Exception as e:
            logger.error(f"Failed to save extraction result for {pdf_path}: {e}")
    
    def extract_single_pdf(self, pdf_path: str) -> Optional[Dict[Any, Any]]:
        """Process single PDF file using GPT-4.1 + Responses API"""
        vector_store = None  # Initialize to track for cleanup
        try:
            # Use the shared client instance
            client = self.client
            
            file_name = os.path.basename(pdf_path)
            logger.info(f"Starting GPT-4.1 processing for PDF: {file_name}")
            
            # Validate PDF file
            is_valid, message = self.validate_pdf_file(pdf_path)
            if not is_valid:
                raise ValueError(f"PDF validation failed: {message}")
            
            logger.info(f"PDF validation successful: {message}")
            
            # Create Vector Store
            vector_store = client.vector_stores.create(name=f"Medical_Literature_{file_name}")
            logger.debug(f"Created Vector Store: {vector_store.id}")
            
            # Upload PDF to Vector Store
            with open(pdf_path, "rb") as file_stream:
                file_batch = client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store.id, 
                    files=[file_stream]
                )
            logger.debug(f"PDF upload complete, status: {file_batch.status}")
            
            # Create medical extraction query
            extraction_query = f"""Extract and catalog relevant information for each patient mentioned in this medical literature.

This is very important to my career. Please ensure that no patient is left out and each patient's data is presented individually.

Please extract the following information:
- Phenotypes: Extract phenotypes/clinical features EXACTLY as described in the original text (no HPO codes, just the raw descriptions from the document)
- Causative_Gene: Identify the gene responsible using the gene symbol
- Causative_Variant: Extract the causative variant using HGVS nomenclature
- Disease: Extract the disease name
- Testing method: What genetic testing was performed (WGS, WES, Panel, CMA, etc.)
- Reference genome and chromosomal coordinates if available

Present your findings in a structured JSON format:

{{
  "document_info": {{
    "file_name": "{file_name}",
    "total_patients_found": <number>,
    "extraction_confidence": "high/medium/low",
    "explanation": "brief explanation of what was found"
  }},
  "patients": [
    {{
      "patient_id": "unique_identifier",
      "demographics": {{
        "age": "age_information",
        "sex": "male/female/unknown",
        "ethnicity": "ethnicity_if_available",
        "nationality": "nationality_if_available",
        "occupation": "occupation_if_available"
      }},
      "phenotypes": [
        "exact phenotype description from the document",
        "another phenotype as written in the paper"
      ],
      "genotype": {{
        "gene": "gene_symbol",
        "testing_method": "WGS/WES/Panel/CMA/Sanger/other (specify if other)",
        "variants": [
          {{
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
          }}
        ]
      }},
      "diagnosis": "disease_name"
    }}
  ]
}}

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
12. For reference_genome and chromosomal_position: Extract these ONLY if explicitly mentioned in the paper"""

            # Use Responses API + gpt-4.1 + file_search
            # Add retry mechanism
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    response = client.responses.create(
                        input=extraction_query,
                        model="gpt-4.1",
                        tools=[{
                            "type": "file_search",
                            "vector_store_ids": [vector_store.id],
                        }]
                    )
                    break  # Success, exit loop
                except Exception as e:
                    if attempt < max_attempts - 1:
                        logger.warning(f"Responses API call failed (attempt {attempt + 1}/{max_attempts}): {e}")
                        time.sleep(5 * (attempt + 1))  # Incremental wait time
                    else:
                        raise
            
            # Extract response content
            response_content = response.output_text if hasattr(response, 'output_text') and response.output_text else str(response.output)
            
            logger.debug(f"Got extraction result, length: {len(response_content)}")
            
            # Try to parse JSON
            try:
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response_content[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    result = {
                        "document_info": {
                            "file_name": file_name,
                            "total_patients_found": 0,
                            "extraction_confidence": "low",
                            "complex_associations": False
                        },
                        "patients": [],
                        "raw_response": response_content
                    }
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed, saving raw response: {e}")
                result = {
                    "document_info": {
                        "file_name": file_name,
                        "total_patients_found": 0,
                        "extraction_confidence": "low",
                        "complex_associations": False
                    },
                    "patients": [],
                    "raw_response": response_content,
                    "json_error": str(e)
                }
            
            # Data structure validation
            is_valid, validation_errors = validate_extraction_result(result)
            result['structure_validation'] = {
                'is_valid': is_valid,
                'errors': validation_errors
            }
            
            # Hallucination detection using markdown content comparison
            pmid = os.path.basename(pdf_path).replace('.pdf', '')
            result['hallucination_check'] = self.hallucination_checker.check_extraction(pmid, result)
            
            # Add processing info
            file_size = os.path.getsize(pdf_path)
            result['processing_info'] = {
                'timestamp': datetime.now().isoformat(),
                'model_used': 'gpt-4.1',
                'api_used': 'responses_api',
                'success': True,
                'extraction_method': 'gpt41_responses_file_search',
                'file_size_kb': round(file_size / 1024, 1),
                'vector_store_id': vector_store.id,
                'usage': getattr(response, 'usage', {}).model_dump() if hasattr(getattr(response, 'usage', {}), 'model_dump') else {}
            }
            
            # Clean up Vector Store
            try:
                client.vector_stores.delete(vector_store.id)
                logger.debug(f"Deleted Vector Store: {vector_store.id}")
            except Exception as e:
                logger.warning(f"Failed to delete Vector Store: {e}")
            
            # Update document_info
            result['document_info']['file_path'] = pdf_path
            result['document_info']['file_name'] = file_name
            
            quality_score = get_quality_score(result)
            logger.info(f"✅ PDF {file_name} GPT-4.1 extraction successful, patients: {len(result.get('patients', []))}, quality: {quality_score:.2f}")
            
            # Save individual result
            self.save_individual_result(result, pdf_path)
            
            return result
                
        except Exception as e:
            logger.error(f"❌ PDF {os.path.basename(pdf_path)} GPT-4.1 extraction failed: {e}")
            error_result = {
                'document_info': {
                    'file_path': pdf_path,
                    'file_name': os.path.basename(pdf_path)
                },
                'patients': [],
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'success': False,
                    'error': str(e),
                    'extraction_method': 'gpt41_responses_file_search',
                    'model_used': 'gpt-4.1',
                    'api_used': 'responses_api'
                }
            }
            
            # Clean up resources
            try:
                if vector_store is not None:
                    client.vector_stores.delete(vector_store.id)
                    logger.debug(f"Cleaned up Vector Store in exception handler: {vector_store.id}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up vector store: {cleanup_error}")
            
            # Save failed individual result
            self.save_individual_result(error_result, pdf_path)
            
            return error_result
    
    def process_pdfs_parallel(
        self,
        pdf_files: List[str],
        max_workers: int = 3,  # Lower concurrency to avoid API limits
        checkpoint_file: str = None
    ) -> List[Dict]:
        """Process multiple PDF files in parallel with checkpoint support"""
        
        # Use default checkpoint file if not specified
        if checkpoint_file is None:
            checkpoint_file = self.checkpoint_file
        
        # Load existing results
        existing_results = self.load_checkpoint(checkpoint_file)
        processed_files = {r['document_info']['file_path'] for r in existing_results}
        
        # Filter out already processed files
        remaining_files = [f for f in pdf_files if f not in processed_files]
        
        logger.info(f"Total PDF files: {len(pdf_files)}")
        logger.info(f"Completed: {len(processed_files)}")
        logger.info(f"Remaining: {len(remaining_files)}")
        
        if not remaining_files:
            logger.info("All PDF files have been processed")
            return existing_results
        
        logger.info(f"Starting parallel processing of {len(remaining_files)} remaining PDF files, max workers: {max_workers}")
        
        results = existing_results.copy()
        completed_count = len(processed_files)
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.extract_single_pdf, pdf_file): pdf_file
                for pdf_file in remaining_files
            }
            
            # Collect results
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="GPT-4.1 processing PDFs"):
                pdf_file = future_to_file[future]
                try:
                    result = future.result(timeout=900)  # 15 minutes timeout
                    results.append(result)
                    completed_count += 1
                    
                    # Save checkpoint every 20 files (reduced frequency)
                    if completed_count % 20 == 0:
                        self.save_checkpoint(results, checkpoint_file)
                        logger.info(f"Checkpoint saved, completed: {completed_count}/{len(pdf_files)}")
                    
                except Exception as e:
                    logger.error(f"❌ PDF {os.path.basename(pdf_file)} processing exception: {e}")
                    error_result = {
                        'document_info': {
                            'file_path': pdf_file,
                            'file_name': os.path.basename(pdf_file)
                        },
                        'patients': [],
                        'processing_info': {
                            'timestamp': datetime.now().isoformat(),
                            'success': False,
                            'error': str(e),
                            'extraction_method': 'gpt5_responses_file_search'
                        }
                    }
                    results.append(error_result)
                    completed_count += 1
                    
                    # Save checkpoint including errors (reduced frequency)
                    if completed_count % 20 == 0:
                        self.save_checkpoint(results, checkpoint_file)
                        logger.info(f"Checkpoint saved (with errors), completed: {completed_count}/{len(pdf_files)}")
            
            # Final checkpoint save
            self.save_checkpoint(results, checkpoint_file)
        
        return results
    
    def save_results_to_csv(self, results: List[Dict], output_path: str):
        """Save results to CSV file"""
        logger.info(f"Saving results to CSV file: {output_path}")
        
        # Build CSV data
        csv_data = []
        
        for result in results:
            doc_info = result.get('document_info', {})
            processing_info = result.get('processing_info', {})
            hallucination_check = result.get('hallucination_check', {})
            
            # Basic info
            row = {
                'file_path': doc_info.get('file_path', ''),
                'file_name': doc_info.get('file_name', ''),
                'extraction_success': processing_info.get('success', False),
                'extraction_timestamp': processing_info.get('timestamp', ''),
                'extraction_method': processing_info.get('extraction_method', 'gpt5_responses_file_search'),
                'model_used': processing_info.get('model_used', 'gpt-5'),
                'api_used': processing_info.get('api_used', 'responses_api'),
                'patients_count': len(result.get('patients', [])),
                'quality_score': get_quality_score(result),
                'credibility_score': hallucination_check.get('credibility_score', 0.9),
                'total_variants': hallucination_check.get('total_variants', 0),
                'verified_variants': hallucination_check.get('verified_variants', 0),
                'suspicious_variants': hallucination_check.get('suspicious_variants', 0),
                'file_size_kb': processing_info.get('file_size_kb', 0),
                'extraction_confidence': doc_info.get('extraction_confidence', 'unknown'),
                'extraction_result': json.dumps(result, ensure_ascii=False)
            }
            
            csv_data.append(row)
        
        # Save CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"CSV saved, total {len(df)} records")
        return df

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-4.1 Responses API Medical Literature PDF Information Extractor")
    parser.add_argument("--input", "-i", required=True, help="PDF file directory path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file path")
    parser.add_argument("--max-files", "-m", type=int, default=20, help="Maximum number of files to process")
    parser.add_argument("--workers", "-w", type=int, default=3, help="Maximum concurrency (recommended 2-4)")
    parser.add_argument("--api-key", "-k", help="OpenAI API key")
    parser.add_argument("--checkpoint", "-c", default="checkpoints/gpt5_responses_results.json", 
                       help="Checkpoint file path")
    parser.add_argument("--keep-individual", action="store_true", 
                       help="Keep detailed result files for each document")
    
    args = parser.parse_args()
    
    try:
        extractor = GPT41ResponsesPDFExtractor(api_key=args.api_key)
        
        # Get PDF file list
        pdf_files = extractor.get_pdf_files(args.input)
        
        if not pdf_files:
            print(f"No PDF files found in directory {args.input}")
            return
        
        # Validate PDF files
        valid_files = []
        for pdf_file in pdf_files:
            is_valid, message = extractor.validate_pdf_file(pdf_file)
            if is_valid:
                valid_files.append(pdf_file)
                logger.info(f"✅ {os.path.basename(pdf_file)}: {message}")
            else:
                logger.warning(f"⚠️  {os.path.basename(pdf_file)}: {message}")
        
        if not valid_files:
            print("No valid PDF files found")
            return
        
        # Limit number of files to process
        if len(valid_files) > args.max_files:
            valid_files = valid_files[:args.max_files]
            print(f"Limited to processing first {args.max_files} valid PDF files")
        
        print(f"Ready to process {len(valid_files)} valid PDF files")
        
        # Parallel processing
        start_time = time.time()
        results = extractor.process_pdfs_parallel(valid_files, args.workers, args.checkpoint)
        end_time = time.time()
        
        # Statistics
        successful = sum(1 for r in results if r.get('processing_info', {}).get('success', False))
        failed = len(results) - successful
        
        print(f"\n🎉 GPT-4.1 Responses API processing complete!")
        print(f"⏱️  Total time: {end_time - start_time:.1f}s")
        print(f"📊 Success: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"❌ Failed: {failed}")
        
        # Save results
        df = extractor.save_results_to_csv(results, args.output)
        
        # Display extraction statistics
        total_patients = sum(r['patients_count'] for _, r in df.iterrows())
        avg_quality = df['quality_score'].mean()
        
        print(f"👥 Total patients extracted: {total_patients}")
        print(f"🎯 Average quality score: {avg_quality:.3f}")
        
        print(f"\n📁 Results saved to: {args.output}")
        print(f"📁 Checkpoint file: {args.checkpoint}")
        
        print(f"\n💡 Comparison analysis command:")
        print(f"python src/comparison_analyzer.py -p {args.output} -m /path/to/markdown_results.tsv -o comparisons/gpt5_vs_markdown")
        
        # Manage individual result files
        try:
            individual_results_count = len([f for f in os.listdir("individual_results") if f.endswith("_gpt5_result.json")])
            print(f"📁 Individual result files: {individual_results_count} (saved in individual_results/ directory)")
            
            if not args.keep_individual and individual_results_count > 0:
                print("💡 To keep individual result files, use --keep-individual parameter")
        except:
            pass
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted, but checkpoint saved!")
        print(f"💡 Resume processing command:")
        print(f"python {__file__} -i {args.input} -o {args.output} -c {args.checkpoint} -m {args.max_files} -w {args.workers}")
    except Exception as e:
        logger.error(f"Processing error: {e}")
        print(f"\n⚠️  Processing error, but checkpoint saved!")
        print(f"💡 Resume processing command:")
        print(f"python {__file__} -i {args.input} -o {args.output} -c {args.checkpoint} -m {args.max_files} -w {args.workers}")
        raise

if __name__ == "__main__":
    main()