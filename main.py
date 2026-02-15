"""
Main entry point for Resume Evaluation System.

Orchestrates:
- Resume ingestion from Google Drive or local storage
- Resume embedding and vector indexing
- Job evaluation with LLM
- Result filtering and delivery
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # File handler
    log_file = Path("logs") / f"app_{Path.cwd().name}.log"
    log_file.parent.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(formatter)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Level: {level}, File: {log_file}")
    
    return logger


def validate_environment() -> Dict[str, str]:
    """
    Validate required environment variables.
    
    Returns:
        Dictionary with validated environment variables
    
    Raises:
        ValueError: If required variables missing
    """
    logger = logging.getLogger(__name__)
    
    required_vars = {
        "GEMINI_API_KEY": "Google Gemini API key",
        "SENDER_EMAIL": "Email sender address",
        "SENDER_PASSWORD": "Email sender password",
        "RECIPIENT_EMAIL": "Email recipient address"
    }
    
    optional_vars = {
        "GDRIVE_CREDENTIALS": "Google Drive credentials file",
        "GDRIVE_RESUMES_FOLDER": "Google Drive resumes folder name",
        "LOG_LEVEL": "Logging level (DEBUG/INFO/WARNING/ERROR)"
    }
    
    env_vars = {}
    missing_vars = []
    
    # Check required variables
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing_vars.append(f"{var} ({description})")
        else:
            env_vars[var] = value
            logger.debug(f"✓ {var} found")
    
    if missing_vars:
        logger.error(f"Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"  - {var}")
        raise ValueError(f"Missing {len(missing_vars)} required environment variables")
    
    # Check optional variables
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            env_vars[var] = value
            logger.debug(f"✓ {var} found (optional)")
        else:
            logger.debug(f"⚠ {var} not set (optional)")
    
    logger.info("✓ Environment validation passed")
    return env_vars


def ingest_resumes_mode(args: argparse.Namespace, env_vars: Dict) -> None:
    """
    Run resume ingestion pipeline.
    
    Args:
        args: Command line arguments
        env_vars: Environment variables
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("RESUME INGESTION MODE")
    logger.info("=" * 60)
    
    try:
        from retrieval.search import ResumeSearchEngine
        
        # Initialize search engine
        logger.info("Initializing search engine...")
        engine = ResumeSearchEngine(
            embedding_model=args.embedding_model,
            dimension=args.embedding_dimension
        )
        
        # Determine resume source
        resume_folder = args.resume_folder or os.getenv(
            "RESUME_FOLDER", 
            "resumes"
        )
        
        # Ingest resumes
        logger.info(f"Ingesting resumes from: {resume_folder}")
        ingest_status = engine.ingest_resumes(
            resume_folder,
            save_index=args.save_index,
            save_path=args.index_path
        )
        
        # Report results
        if ingest_status["success"]:
            logger.info("✓ Ingestion successful")
            logger.info(f"  Resumes ingested: {ingest_status['resumes_ingested']}")
            logger.info(f"  Parse stats: {ingest_status['parse_stats']}")
            logger.info(f"  Index stats: {ingest_status['index_stats']}")
        else:
            logger.error(f"✗ Ingestion failed: {ingest_status.get('error')}")
    
    except Exception as e:
        logger.error(f"Ingestion mode failed: {e}", exc_info=True)
        sys.exit(1)


def evaluate_mode(args: argparse.Namespace, env_vars: Dict) -> None:
    """
    Run resume evaluation pipeline.
    
    Args:
        args: Command line arguments
        env_vars: Environment variables
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("RESUME EVALUATION MODE")
    logger.info("=" * 60)
    
    try:
        from agent import Agent
        
        # Get job description
        if args.job_file:
            logger.info(f"Reading job description from: {args.job_file}")
            with open(args.job_file, 'r') as f:
                job_description = f.read()
            logger.debug(f"Job description loaded ({len(job_description)} chars)")
        else:
            job_description = args.job or (
                "Senior Backend Engineer with Python, AWS, and microservices"
            )
            logger.info(f"Using provided job description: {job_description[:50]}...")
        
        # Create agent
        logger.info("Initializing agent...")
        agent = Agent()
        
        # Run evaluation
        logger.info("Starting evaluation...")
        query = args.query or "top 10"
        result = agent.run(
            job_description=job_description,
            query=query
        )
        
        # Report results
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        
        if result["success"]:
            logger.info("✓ Evaluation successful")
            summary = result.get("summary", {})
            logger.info(f"  Resumes loaded: {summary.get('resumes_loaded')}")
            logger.info(f"  Candidates evaluated: {summary.get('candidates_evaluated')}")
        else:
            logger.error(f"✗ Evaluation failed: {result.get('error')}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Evaluation mode failed: {e}", exc_info=True)
        sys.exit(1)


def batch_mode(args: argparse.Namespace, env_vars: Dict) -> None:
    """
    Run batch evaluation against multiple jobs.
    
    Args:
        args: Command line arguments
        env_vars: Environment variables
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("BATCH EVALUATION MODE")
    logger.info("=" * 60)
    
    try:
        import json
        from agent import Agent
        
        # Read batch file
        if not args.batch_file:
            logger.error("Batch file required (--batch-file)")
            sys.exit(1)
        
        logger.info(f"Reading batch jobs from: {args.batch_file}")
        with open(args.batch_file, 'r') as f:
            batch_jobs = json.load(f)
        
        logger.info(f"Loaded {len(batch_jobs)} jobs")
        
        # Create agent
        agent = Agent()
        
        # Run batch evaluation
        logger.info("Starting batch evaluation...")
        result = agent.run(batch_jobs=batch_jobs)
        
        # Report results
        logger.info("=" * 60)
        logger.info("BATCH RESULTS")
        logger.info("=" * 60)
        
        if result["success"]:
            logger.info("✓ Batch evaluation complete")
            logger.info(f"  Total jobs: {result.get('total_jobs')}")
            logger.info(f"  Successful: {result.get('successful')}")
            logger.info(f"  Failed: {result.get('failed')}")
            
            # Save results
            output_file = args.output_file or "batch_results.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to: {output_file}")
        else:
            logger.error(f"✗ Batch evaluation failed: {result.get('error')}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Batch mode failed: {e}", exc_info=True)
        sys.exit(1)


def demo_mode(args: argparse.Namespace, env_vars: Dict) -> None:
    """
    Run demonstration with sample data.
    
    Args:
        args: Command line arguments
        env_vars: Environment variables
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("DEMO MODE")
    logger.info("=" * 60)
    
    try:
        from agent import Agent
        
        # Sample job descriptions
        jobs = [
            {
                "id": "job_1",
                "title": "Senior Python Engineer",
                "description": """
                Position: Senior Python Engineer
                
                Requirements:
                - 5+ years Python experience
                - Strong microservices architecture knowledge
                - AWS/GCP cloud platform expertise
                - SQL database design
                - Team leadership experience
                
                Nice to have:
                - Machine learning experience
                - Kubernetes knowledge
                - Open source contributions
                """
            },
            {
                "id": "job_2",
                "title": "Data Science Engineer",
                "description": """
                Position: Data Science Engineer
                
                Requirements:
                - 3+ years data science experience
                - Python, SQL proficiency
                - ML model development
                - Big data tools (Spark, Hadoop)
                - Statistical analysis
                """
            }
        ]
        
        # Create agent
        logger.info("Initializing agent...")
        agent = Agent()
        
        # Run batch demo
        logger.info("Running demo with sample jobs...")
        result = agent.run(batch_jobs=jobs)
        
        # Report results
        logger.info("=" * 60)
        logger.info("DEMO RESULTS")
        logger.info("=" * 60)
        
        if result["success"]:
            logger.info("✓ Demo completed successfully")
            logger.info(f"  Jobs processed: {result.get('successful')}/{result.get('total_jobs')}")
        else:
            logger.info(f"⚠ Demo completed with note: {result.get('error')}")
    
    except Exception as e:
        logger.error(f"Demo mode failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main application entry point."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Resume Evaluation System - AI-powered resume screening"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["ingest", "evaluate", "batch", "demo"],
        default="evaluate",
        help="Execution mode (default: evaluate)"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    # Ingestion mode arguments
    parser.add_argument(
        "--resume-folder",
        help="Path to resume folder (ingestion mode)"
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=384,
        help="Embedding dimension (default: 384)"
    )
    parser.add_argument(
        "--save-index",
        action="store_true",
        help="Save index after ingestion"
    )
    parser.add_argument(
        "--index-path",
        default="vector_store",
        help="Path to save/load index (default: vector_store)"
    )
    
    # Evaluation mode arguments
    parser.add_argument(
        "--job",
        help="Job description text (evaluation mode)"
    )
    parser.add_argument(
        "--job-file",
        help="Path to job description file (evaluation mode)"
    )
    parser.add_argument(
        "--query",
        help="Query to filter results (evaluation mode)"
    )
    
    # Batch mode arguments
    parser.add_argument(
        "--batch-file",
        help="Path to batch jobs JSON file (batch mode)"
    )
    parser.add_argument(
        "--output-file",
        help="Output file for batch results (batch mode)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("Resume Evaluation System")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Log level: {args.log_level}")
    
    try:
        # Validate environment
        logger.info("Validating environment...")
        env_vars = validate_environment()
        
        # Route to appropriate mode
        if args.mode == "ingest":
            ingest_resumes_mode(args, env_vars)
        
        elif args.mode == "evaluate":
            evaluate_mode(args, env_vars)
        
        elif args.mode == "batch":
            batch_mode(args, env_vars)
        
        elif args.mode == "demo":
            demo_mode(args, env_vars)
        
        logger.info("=" * 60)
        logger.info("✓ Application completed successfully")
        logger.info("=" * 60)
    
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()