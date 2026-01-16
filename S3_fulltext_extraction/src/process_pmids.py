#!/usr/bin/env python3
"""
Main script to process PMIDs from CSV file with checkpoint support
"""
import os
import sys
import csv
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

# Ensure src directory is importable when executed as a script
SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from gpt41_responses_pdf_extractor import GPT41ResponsesPDFExtractor
from config import config

# Configure logging (logs stored at project root level)
log_dir = SRC_ROOT.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f'process_pmids_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Log file: {log_file}")

class PMIDProcessor:
    """Process PMIDs from CSV with checkpoint support"""
    
    def __init__(self, 
                 csv_file: str,
                 output_dir: str = "extraction_results",
                 checkpoint_file: str = "checkpoint.json",
                 pdf_dir: str = None):
        """
        Initialize processor
        
        Args:
            csv_file: Path to CSV file containing PMIDs
            output_dir: Directory to save extracted JSON files
            checkpoint_file: File to track processing progress
            pdf_dir: Directory containing PDF files
        """
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.checkpoint_file = checkpoint_file
        # Use config PDF directory if not specified
        self.pdf_dir = pdf_dir if pdf_dir is not None else config.pdf_directory
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize extractor
        self.extractor = GPT41ResponsesPDFExtractor()
        
        # Load or initialize checkpoint
        self.checkpoint = self.load_checkpoint()
        
        # Thread lock for checkpoint updates
        self.checkpoint_lock = threading.Lock()
        
    def load_checkpoint(self) -> Dict:
        """Load checkpoint from file or create new one"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                logger.info(f"Loaded checkpoint: {checkpoint['stats']}")
                return checkpoint
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, creating new one")
        
        return {
            "processed": {},  # pmid -> status
            "stats": {
                "total": 0,
                "success": 0,
                "failed": 0,
                "skipped": 0
            },
            "last_updated": None
        }
    
    def save_checkpoint(self):
        """Save checkpoint to file (thread-safe with retry)"""
        with self.checkpoint_lock:
            self.checkpoint["last_updated"] = datetime.now().isoformat()
            
            # Retry mechanism to handle file handle issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(self.checkpoint_file, 'w') as f:
                        json.dump(self.checkpoint, f, indent=2)
                    break  # Success, exit retry loop
                except (OSError, IOError) as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to save checkpoint after {max_retries} attempts: {e}")
                        # Don't raise, just log the error
                        break
                    else:
                        logger.warning(f"Checkpoint save attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(0.1)  # Short delay before retry
            
    def read_pmids_from_csv(self) -> List[str]:
        """Read PMIDs from CSV file"""
        pmids = []
        
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check for pmid column (case-insensitive)
            if reader.fieldnames:
                pmid_col = None
                for col in reader.fieldnames:
                    if col.lower() == 'pmid':
                        pmid_col = col
                        break
                
                if not pmid_col:
                    raise ValueError(f"No 'pmid' column found in CSV. Available columns: {reader.fieldnames}")
                
                for row in reader:
                    pmid = row[pmid_col].strip()
                    if pmid:
                        pmids.append(pmid)
        
        logger.info(f"Found {len(pmids)} PMIDs in CSV")
        return pmids
    
    def process_single_pmid(self, pmid: str) -> Dict:
        """Process a single PMID"""
        result = {
            "pmid": pmid,
            "status": "pending",
            "error": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            output_file = os.path.join(self.output_dir, f"{pmid}.json")

            # Check if already processed (with lock to avoid race condition)
            with self.checkpoint_lock:
                entry = self.checkpoint["processed"].get(pmid)
                entry_status = entry.get("status") if entry else None
                existing_json = os.path.exists(output_file)

                if existing_json and entry_status != "success":
                    # Recover from missing checkpoint entry by registering success.
                    self.checkpoint["processed"][pmid] = {
                        "pmid": pmid,
                        "status": "success",
                        "output_file": output_file,
                        "timestamp": entry.get("timestamp") if entry else datetime.now().isoformat(),
                        "patients_found": entry.get("patients_found", 0) if entry else 0,
                    }
                    # Defer saving to avoid excessive I/O; main loop will flush periodically.
                    entry_status = "success"

                if entry_status == "success" and existing_json:
                    logger.info(f"Skipping {pmid} - already processed successfully")
                    result["status"] = "success"
                    result["output_file"] = output_file
                    result["patients_found"] = entry.get("patients_found", 0) if entry else 0
                    return result

            # Construct PDF path
            pdf_path = os.path.join(self.pdf_dir, f"{pmid}.pdf")
            
            # Check if PDF exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            logger.info(f"Processing {pmid} from {pdf_path}")
            
            # Extract information
            extraction_result = self.extractor.extract_single_pdf(pdf_path)
            
            if extraction_result:
                # Save to output file
                output_file = os.path.join(self.output_dir, f"{pmid}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(extraction_result, f, ensure_ascii=False, indent=2)
                
                # Check if extraction was actually successful
                processing_info = extraction_result.get("processing_info", {})
                if processing_info.get("success", False):
                    logger.info(f"✅ Successfully processed {pmid} -> {output_file}")
                    result["status"] = "success"
                    result["output_file"] = output_file
                    result["patients_found"] = len(extraction_result.get("patients", []))
                else:
                    # Extraction failed (e.g., timeout)
                    error_msg = processing_info.get("error", "Unknown error")
                    logger.error(f"❌ Extraction failed for {pmid}: {error_msg}")
                    result["status"] = "failed"
                    result["error"] = error_msg
                    result["output_file"] = output_file
            else:
                raise ValueError("Extraction returned empty result")
                
        except FileNotFoundError as e:
            logger.error(f"❌ PDF not found for {pmid}: {e}")
            result["status"] = "failed"
            result["error"] = f"PDF not found: {str(e)}"
            
        except Exception as e:
            logger.exception(f"❌ Failed to process {pmid}: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            
        return result
    
    def process_all(self, max_workers: int = 5, retry_failed: bool = False):
        """
        Process all PMIDs from CSV with parallel processing
        
        Args:
            max_workers: Number of concurrent workers (default 5)
            retry_failed: Whether to retry previously failed PMIDs
        """
        # Read PMIDs from CSV
        pmids = self.read_pmids_from_csv()
        self.checkpoint["stats"]["total"] = len(pmids)
        
        logger.info("="*60)
        logger.info(f"📚 Starting batch processing of {len(pmids)} PMIDs")
        logger.info(f"🔧 Using {max_workers} parallel workers")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"💾 Checkpoint file: {self.checkpoint_file}")
        logger.info("="*60)
        
        # Filter PMIDs to process
        pmids_to_process = []
        skipped_success = 0
        skipped_failed = 0
        retry_count = 0  # 记录重试的失败数
        
        for pmid in pmids:
            with self.checkpoint_lock:
                if pmid in self.checkpoint["processed"]:
                    status = self.checkpoint["processed"][pmid]["status"]
                    if status == "success":
                        skipped_success += 1
                        if not retry_failed:
                            self.checkpoint["stats"]["skipped"] += 1
                        continue
                    elif status == "failed":
                        if retry_failed:
                            # 在retry模式下，失败的会被重新处理
                            retry_count += 1
                            pmids_to_process.append(pmid)
                        else:
                            # 非retry模式，跳过失败的
                            skipped_failed += 1
                            self.checkpoint["stats"]["skipped"] += 1
                            continue
                else:
                    # 未处理过的PMID
                    pmids_to_process.append(pmid)
        
        # 根据模式打印不同的汇总信息
        if retry_failed:
            logger.info(f"📊 跳过{skipped_success}个已成功，准备重试{retry_count}个失败的PMID")
            logger.info(f"🔄 重试模式：将处理{len(pmids_to_process)}个PMID（{retry_count}个失败重试 + {len(pmids_to_process)-retry_count}个未处理）")
        else:
            if skipped_success > 0 or skipped_failed > 0:
                logger.info(f"📊 跳过已处理: {skipped_success}个成功, {skipped_failed}个失败")
            logger.info(f"📋 将处理{len(pmids_to_process)}个未处理的PMID")
        
        if not pmids_to_process:
            logger.info("✅ 没有需要处理的PMID")
            return
        
        logger.info(f"\n📋 开始处理 {len(pmids_to_process)} 个PMID...")
        
        # Process PMIDs in parallel with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pmid = {
                executor.submit(self.process_single_pmid, pmid): pmid 
                for pmid in pmids_to_process
            }
            
            # Initialize progress bar
            pbar = tqdm(
                total=len(pmids_to_process),
                desc="Processing PMIDs",
                unit="PMIDs",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            
            # Process completed tasks
            completed = 0
            success_count = 0
            failed_count = 0
            skipped_count = 0
            
            for future in as_completed(future_to_pmid):
                pmid = future_to_pmid[future]
                completed += 1
                
                try:
                    result = future.result()
                    
                    # Update checkpoint (thread-safe)
                    with self.checkpoint_lock:
                        # 不保存skipped状态到checkpoint，避免覆盖真实的处理结果
                        if result["status"] != "skipped":
                            prev_status = self.checkpoint["processed"].get(pmid, {}).get("status")
                            self.checkpoint["processed"][pmid] = result
                        
                        # Update statistics
                        if result["status"] == "success":
                            if prev_status != "success":
                                self.checkpoint["stats"]["success"] += 1
                                success_count += 1
                            pbar.set_postfix({
                                "✅": success_count,
                                "❌": failed_count,
                                "⏭️": skipped_count,
                                "Current": pmid
                            })
                        elif result["status"] == "failed":
                            self.checkpoint["stats"]["failed"] += 1
                            failed_count += 1
                            pbar.set_postfix({
                                "✅": success_count,
                                "❌": failed_count,
                                "⏭️": skipped_count,
                                "Current": f"{pmid} (FAILED)"
                            })
                        elif result["status"] == "skipped":
                            self.checkpoint["stats"]["skipped"] += 1
                            skipped_count += 1
                            pbar.set_postfix({
                                "✅": success_count,
                                "❌": failed_count,
                                "⏭️": skipped_count,
                                "Current": f"{pmid} (SKIPPED)"
                            })
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Save checkpoint periodically (reduced frequency)
                    if completed % 20 == 0 or completed == len(pmids_to_process):
                        self.save_checkpoint()
                        
                except Exception as e:
                    failed_count += 1
                    pbar.set_postfix({
                        "✅": success_count,
                        "❌": failed_count,
                        "⏭️": skipped_count,
                        "Current": f"{pmid} (ERROR)"
                    })
                    pbar.update(1)
                    with self.checkpoint_lock:
                        self.checkpoint["processed"][pmid] = {
                            "pmid": pmid,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        self.checkpoint["stats"]["failed"] += 1
                    # Only save checkpoint for failures every 10 failures to reduce I/O
                    if failed_count % 10 == 0:
                        self.save_checkpoint()
            
            # Close progress bar
            pbar.close()
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("📊 Processing Complete!")
        logger.info(f"   Total: {self.checkpoint['stats']['total']}")
        logger.info(f"   ✅ Success: {self.checkpoint['stats']['success']}")
        logger.info(f"   ❌ Failed: {self.checkpoint['stats']['failed']}")
        logger.info(f"   ⏭️  Skipped: {self.checkpoint['stats']['skipped']}")
        logger.info("="*60)
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate a summary report of processing results"""
        report_file = os.path.join(self.output_dir, "processing_summary.json")
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "stats": self.checkpoint["stats"],
            "successful_pmids": [],
            "failed_pmids": []
        }
        
        for pmid, info in self.checkpoint["processed"].items():
            if info["status"] == "success":
                summary["successful_pmids"].append({
                    "pmid": pmid,
                    "output_file": info.get("output_file"),
                    "patients_found": info.get("patients_found", 0)
                })
            elif info["status"] == "failed":
                summary["failed_pmids"].append({
                    "pmid": pmid,
                    "error": info.get("error")
                })
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 Summary report saved to: {report_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PMIDs from CSV file")
    parser.add_argument("csv_file", help="Path to CSV file containing PMIDs")
    parser.add_argument("--output-dir", "-o", default="extraction_results",
                       help="Output directory for JSON files (default: extraction_results)")
    parser.add_argument("--checkpoint", "-c", default="checkpoint.json",
                       help="Checkpoint file for tracking progress (default: checkpoint.json)")
    parser.add_argument("--pdf-dir", "-p", default=None,
                       help="Directory containing PDF files (default: use config.PDF_DIRECTORY)")
    parser.add_argument("--retry-failed", action="store_true",
                       help="Retry previously failed PMIDs")
    parser.add_argument("--reset", action="store_true",
                       help="Reset checkpoint and start fresh")
    parser.add_argument("--workers", "-w", type=int, default=None,
                       help="Number of parallel workers (default: use config.MAX_WORKERS)")
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    # Reset checkpoint if requested
    if args.reset and os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        logger.info("Checkpoint reset")
    
    # Use config PDF directory if not specified via command line
    pdf_dir = args.pdf_dir if args.pdf_dir is not None else config.pdf_directory
    
    # Create processor and run
    processor = PMIDProcessor(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        checkpoint_file=args.checkpoint,
        pdf_dir=pdf_dir
    )
    
    # Use command line workers or config default
    workers = args.workers if args.workers is not None else config.max_workers
    logger.info(f"Using {workers} parallel workers")
    
    processor.process_all(max_workers=workers, retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
