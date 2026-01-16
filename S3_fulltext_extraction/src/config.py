"""
Configuration management module for Medical PDF Extraction Project
"""
import os
from dotenv import load_dotenv
from typing import Optional
from pathlib import Path

class Config:
    """Configuration management class"""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration"""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()  # Load .env file
        
        # Set default project root
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key"""
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError(
                "OpenAI API key not set. Please set OPENAI_API_KEY environment variable or configure in .env file"
            )
        return key
    
    @property
    def openai_model(self) -> str:
        """Get OpenAI model name"""
        return os.getenv("OPENAI_MODEL", "gpt-4.1")
    
    @property
    def openai_temperature(self) -> float:
        """Get temperature parameter"""
        return float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    
    @property
    def openai_max_tokens(self) -> int:
        """Get max tokens"""
        return int(os.getenv("OPENAI_MAX_TOKENS", "8192"))
    
    @property
    def chunk_size(self) -> int:
        """Get batch processing size"""
        return int(os.getenv("CHUNK_SIZE", "100"))
    
    @property
    def max_retries(self) -> int:
        """Get max retry attempts"""
        return int(os.getenv("MAX_RETRIES", "3"))
    
    @property
    def sleep_between_requests(self) -> float:
        """Get sleep time between requests"""
        return float(os.getenv("SLEEP_BETWEEN_REQUESTS", "0.1"))
    
    @property
    def pdf_directory(self) -> str:
        """Get PDF source directory path"""
        # Default to /data/Diseases/pdf/ if not specified
        default_path = "/data/Diseases/pdf"
        pdf_dir = os.getenv("PDF_DIRECTORY", default_path)
        
        # Ensure the directory exists
        if not os.path.exists(pdf_dir):
            print(f"Warning: PDF directory does not exist: {pdf_dir}")
            print(f"Please create it or set PDF_DIRECTORY in .env file")
        
        return pdf_dir
    
    @property
    def extraction_output_dir(self) -> str:
        """Get extraction output directory"""
        # Store extraction results in project directory
        output_dir = os.getenv("EXTRACTION_OUTPUT_DIR", "extraction_results")
        
        # If relative path, make it relative to project root
        if not os.path.isabs(output_dir):
            output_dir = str(self.project_root / output_dir)
        
        return output_dir
    
    @property
    def checkpoint_file(self) -> str:
        """Get checkpoint file path"""
        # Store checkpoint in project root
        return os.getenv("CHECKPOINT_FILE", str(self.project_root / "checkpoint.json"))
    
    @property
    def log_level(self) -> str:
        """Get log level"""
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def max_workers(self) -> int:
        """Get max parallel workers for processing"""
        return int(os.getenv("MAX_WORKERS", "3"))
    
    @property
    def proxy_enabled(self) -> bool:
        """Check if proxy is enabled"""
        return os.getenv("PROXY_ENABLED", "true").lower() == "true"
    
    @property
    def http_proxy(self) -> str:
        """Get HTTP proxy"""
        return os.getenv("HTTP_PROXY", "http://10.152.65.173:7890")
    
    @property
    def https_proxy(self) -> str:
        """Get HTTPS proxy"""
        return os.getenv("HTTPS_PROXY", "http://10.152.65.173:7890")
    
    @property
    def markdown_tsv_path(self) -> str:
        """Get markdown content TSV file path"""
        # Default to /data/markdown_content.tsv if not specified
        default_path = "/data/markdown_content.tsv"
        tsv_path = os.getenv("MARKDOWN_TSV_PATH", default_path)
        
        # Check if file exists
        if not os.path.exists(tsv_path):
            print(f"Warning: Markdown TSV file does not exist: {tsv_path}")
            print(f"Please ensure the file exists or set MARKDOWN_TSV_PATH in .env file")
        
        return tsv_path
    
    def get_openai_config(self) -> dict:
        """Get OpenAI API configuration"""
        return {
            "model": self.openai_model,
            "max_tokens": self.openai_max_tokens,
            "temperature": self.openai_temperature,
            "top_p": 0.9,
            "response_format": {"type": "json_object"}
        }
    
    def override_api_key(self, api_key: str):
        """Override API key (for command line arguments)"""
        os.environ["OPENAI_API_KEY"] = api_key
    
    def get_pdf_path(self, pmid: str) -> str:
        """Get full path for a PDF file given its PMID"""
        return os.path.join(self.pdf_directory, f"{pmid}.pdf")
    
    def ensure_directories(self):
        """Ensure necessary directories exist"""
        # Only create extraction output directory
        os.makedirs(self.extraction_output_dir, exist_ok=True)
        
        # Check if PDF directory exists (but don't create it)
        if not os.path.exists(self.pdf_directory):
            print(f"⚠️  Warning: PDF directory not found: {self.pdf_directory}")
            print(f"   Please ensure PDFs are placed in this directory")

# Global configuration instance
config = Config()