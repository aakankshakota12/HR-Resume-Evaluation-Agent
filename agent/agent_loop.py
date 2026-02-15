"""Agent control loop with state machine and batch processing."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.state import AgentState, StateContext
from agent.memory import AgentMemory, JobMemory, Resume, CandidateEvaluation, EvaluationScores

logger = logging.getLogger(__name__)


class Agent:
    """
    Autonomous Resume Evaluation Agent.
    
    Orchestrates resume ingestion, LLM evaluation, filtering, and result delivery.
    
    State Machine:
        INIT → JOB_SETUP → EVALUATION → QUERY
    """
    
    # Configuration constants
    DEFAULT_SKILL_WEIGHT = 0.5
    DEFAULT_PROJECT_WEIGHT = 0.3
    DEFAULT_CLARITY_WEIGHT = 0.2
    
    def __init__(self):
        """Initialize the agent with clean state."""
        self.memory = AgentMemory()
        self.context = StateContext()
        logger.info("Agent initialized")
    
    def run(self, 
            job_description: str = None, 
            batch_jobs: List[Dict] = None, 
            query: str = None) -> Dict:
        """
        Main entry point for agent execution.
        
        Supports three modes:
        1. Single job: Evaluate resumes against one job
        2. Batch mode: Evaluate resumes against multiple jobs
        3. Query mode: Filter results based on recruiter query
        
        Args:
            job_description: Single job posting text
            batch_jobs: List of jobs [{id, title, description}, ...]
            query: Recruiter query for filtering results
        
        Returns:
            Dictionary with execution results:
            {
                "success": bool,
                "error": str or None,
                "state": current state,
                "summary": execution summary
            }
        
        Raises:
            Exception: Captured and returned in result dict
        """
        try:
            # Batch mode
            if batch_jobs:
                return self._run_batch(batch_jobs)
            
            # Single job mode - setup initial state
            if job_description:
                self._validate_job_description(job_description)
                self.context.job_description = job_description
                self.context.transition_to(AgentState.JOB_SETUP)
            
            if query:
                self._validate_query(query)
                self.context.query = query
                self.context.transition_to(AgentState.QUERY)
            
            # Execute state machine
            max_iterations = 10  # Safety limit to prevent infinite loops
            iterations = 0
            
            while iterations < max_iterations:
                iterations += 1
                state = self.context.current_state
                
                try:
                    if state == AgentState.INIT:
                        self._handle_init()
                    
                    elif state == AgentState.JOB_SETUP:
                        self._handle_job_setup()
                    
                    elif state == AgentState.EVALUATION:
                        self._handle_evaluation()
                    
                    elif state == AgentState.QUERY:
                        self._handle_query()
                        break
                
                except Exception as e:
                    self.context.set_error(str(e))
                    logger.error(f"State {state.value} failed: {e}", exc_info=True)
                    break
            
            if iterations >= max_iterations:
                logger.error("State machine exceeded max iterations")
                self.context.set_error("State machine loop exceeded maximum iterations")
            
            return self._get_result()
        
        except Exception as e:
            logger.error(f"Agent run failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "state": "FAILED",
                "summary": self.memory.get_evaluation_summary()
            }
    
    def _validate_job_description(self, job_description: str) -> None:
        """
        Validate job description input.
        
        Args:
            job_description: Job description text
        
        Raises:
            TypeError: If not string
            ValueError: If empty
        """
        if not isinstance(job_description, str):
            raise TypeError(f"job_description must be string, got {type(job_description)}")
        
        if not job_description.strip():
            raise ValueError("job_description cannot be empty")
    
    def _validate_query(self, query: str) -> None:
        """
        Validate query input.
        
        Args:
            query: Query text
        
        Raises:
            TypeError: If not string
            ValueError: If empty
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be string, got {type(query)}")
        
        if not query.strip():
            raise ValueError("query cannot be empty")
    
    def _handle_init(self) -> None:
        """
        INIT state: Load resumes from source.
        
        Loads resumes from configured source (Google Drive, local, etc).
        Updates state to JOB_SETUP on success.
        
        Raises:
            Exception: If resume loading fails
        """
        logger.info("State: INIT - Loading resumes from source...")
        
        try:
            from tools.resume_source import get_resume_source
            
            source = get_resume_source()
            resumes_data = source.load_resumes()
            
            if not resumes_data:
                raise ValueError("No resumes loaded from source. Check resume folder.")
            
            logger.info(f"Processing {len(resumes_data)} resumes from source...")
            
            for idx, resume_data in enumerate(resumes_data, start=1):
                try:
                    resume = Resume(
                        resume_id=resume_data["id"],
                        candidate_name=resume_data["name"],
                        raw_text=resume_data["text"]
                    )
                    self.memory.resume_memory.add_resume(resume)
                except (KeyError, ValueError) as e:
                    logger.error(f"Invalid resume data at index {idx}: {e}")
                    continue
            
            loaded_count = self.memory.resume_memory.count()
            if loaded_count == 0:
                raise ValueError("No valid resumes loaded after processing")
            
            logger.info(f"✓ Successfully loaded {loaded_count} resumes")
        
        except Exception as e:
            logger.error(f"Resume loading failed: {e}", exc_info=True)
            raise Exception(f"Failed to load resumes: {e}")
        
        self.context.transition_to(AgentState.JOB_SETUP)
    
    def _handle_job_setup(self) -> None:
        """
        JOB_SETUP state: Validate and store job description.
        
        Extracts job title and stores job context.
        Updates state to EVALUATION on success.
        
        Raises:
            ValueError: If job description missing or invalid
        """
        logger.info("State: JOB_SETUP - Preparing job context...")
        
        if not self.context.job_description:
            raise ValueError("Job description required for setup")
        
        if not self.memory.has_resumes():
            raise ValueError("No resumes loaded. Cannot setup job without resumes.")
        
        try:
            # Extract title from first line or use default
            lines = self.context.job_description.split('\n')
            job_title = (lines[0].strip() if lines and lines[0].strip() 
                        else "Untitled Position")
            
            job_memory = JobMemory(
                job_title=job_title,
                job_description=self.context.job_description
            )
            
            self.memory.start_new_job_session(job_memory)
            
            logger.info(f"✓ Job context prepared: {job_memory.job_title}")
            logger.debug(f"Job description length: {len(self.context.job_description)} chars")
        
        except Exception as e:
            logger.error(f"Job setup failed: {e}", exc_info=True)
            raise ValueError(f"Failed to setup job: {e}")
        
        self.context.transition_to(AgentState.EVALUATION)
    
    def _handle_evaluation(self) -> None:
        """
        EVALUATION state: Evaluate all resumes against job.
        
        Calls LLM to evaluate each resume, computes weighted scores.
        Updates state to QUERY on success.
        
        Raises:
            Exception: If evaluation fails
        """
        logger.info("State: EVALUATION - Starting resume evaluation...")
        
        if not self.memory.has_job():
            raise ValueError("Job not loaded. Cannot evaluate without job context.")
        
        try:
            from scoring.llm_scoring import evaluate_resume
            from scoring.rubric import WEIGHTS
        except ImportError as e:
            logger.error(f"Failed to import scoring modules: {e}")
            raise ImportError(f"Missing scoring module: {e}")
        
        resumes = self.memory.resume_memory.get_all_resumes()
        total_resumes = len(resumes)
        logger.info(f"Evaluating {total_resumes} candidates...")
        
        successful_evals = 0
        failed_evals = 0
        failed_candidates = []
        
        for idx, resume in enumerate(resumes, start=1):
            try:
                logger.debug(
                    f"[{idx}/{total_resumes}] Evaluating: {resume.candidate_name}"
                )
                
                # YELLOW LINE 1 - FIX: Added try-catch for LLM call
                try:
                    evaluation_data = evaluate_resume(
                        resume_text=resume.raw_text,
                        job_description=self.memory.job_memory.job_description,
                        job_title=self.memory.job_memory.job_title
                    )
                except Exception as llm_error:
                    logger.warning(
                        f"LLM evaluation failed for {resume.candidate_name}: {llm_error}"
                    )
                    # Use default scores if LLM fails
                    evaluation_data = self._get_default_evaluation()
                
                # Validate LLM response
                self._validate_evaluation_data(evaluation_data)
                
                # Create scores with error handling
                try:
                    scores = EvaluationScores(
                        skill_score=float(evaluation_data["skill_score"]),
                        project_score=float(evaluation_data["project_score"]),
                        clarity_score=float(evaluation_data["clarity_score"])
                    )
                except (ValueError, TypeError) as score_error:
                    logger.warning(f"Score conversion failed: {score_error}")
                    failed_evals += 1
                    failed_candidates.append(resume.candidate_name)
                    continue
                
                # Compute final score with weights from config
                weights = WEIGHTS or {
                    "skill": self.DEFAULT_SKILL_WEIGHT,
                    "project": self.DEFAULT_PROJECT_WEIGHT,
                    "clarity": self.DEFAULT_CLARITY_WEIGHT
                }
                
                scores.compute_final_score(
                    skill_weight=weights.get("skill", self.DEFAULT_SKILL_WEIGHT),
                    project_weight=weights.get("project", self.DEFAULT_PROJECT_WEIGHT),
                    clarity_weight=weights.get("clarity", self.DEFAULT_CLARITY_WEIGHT)
                )
                
                # Create evaluation object
                evaluation = CandidateEvaluation(
                    resume_id=resume.resume_id,
                    candidate_name=resume.candidate_name,
                    scores=scores,
                    strengths=evaluation_data.get("strengths", []),
                    weaknesses=evaluation_data.get("weaknesses", []),
                    skills=evaluation_data.get("skills", []),
                    experience=evaluation_data.get("experience", ""),
                    job_titles=evaluation_data.get("job_titles", [])
                )
                
                self.memory.candidate_memory.add_evaluation(evaluation)
                successful_evals += 1
                logger.debug(
                    f"  ✓ {resume.candidate_name}: "
                    f"Score={scores.final_score:.2f}"
                )
            
            except Exception as e:
                failed_evals += 1
                failed_candidates.append(resume.candidate_name)
                logger.error(
                    f"Evaluation failed for {resume.candidate_name}: {e}"
                )
                continue
        
        # Rank candidates
        ranked = self.memory.candidate_memory.rank_candidates()
        
        # Log summary
        logger.info(
            f"✓ Evaluation complete. "
            f"Success: {successful_evals}/{total_resumes}, "
            f"Failed: {failed_evals}"
        )
        
        if failed_candidates:
            logger.warning(f"Failed candidates: {', '.join(failed_candidates)}")
        
        if ranked:
            top = ranked[0]
            logger.info(
                f"Top candidate: {top.candidate_name} "
                f"(Score: {top.scores.final_score:.2f})"
            )
        else:
            logger.warning("No candidates successfully evaluated")
        
        self.context.transition_to(AgentState.QUERY)
    
    def _handle_query(self) -> None:
        """
        QUERY state: Process recruiter query and send results.
        
        Interprets query intent, filters candidates, sends email.
        Ends agent execution on completion.
        
        Logs warning if no query provided but doesn't fail.
        """
        logger.info("State: QUERY - Processing recruiter query...")
        
        if not self.context.query:
            logger.info("No query provided, skipping query processing")
            return
        
        if not self.memory.candidate_memory.count():
            logger.warning("No candidates evaluated, cannot process query")
            return
        
        try:
            from profiling.query_parser import interpret_query
            from ui.topk import apply_filters
            from tools.email_sender import send_email_results
            
            # YELLOW LINE 2 - FIX: Added error handling for query interpretation
            try:
                query_intent = interpret_query(self.context.query)
                logger.info(f"Query intent: {query_intent.get('intent', 'unknown')}")
            except Exception as query_error:
                logger.warning(f"Query interpretation failed: {query_error}")
                # Use default intent (all candidates)
                query_intent = {
                    "intent": "all",
                    "k": None,
                    "filters": {},
                    "order": "desc",
                    "strict": False
                }
            
            # Get all candidates
            all_candidates = self.memory.candidate_memory.get_all_evaluations()
            
            # Convert to dict format for filtering
            candidates_dict = self._evaluations_to_dict(all_candidates)
            
            # YELLOW LINE 3 - FIX: Added error handling for filtering
            try:
                filtered = apply_filters(candidates_dict, query_intent)
                logger.info(f"✓ Filtered to {len(filtered)} candidates")
            except Exception as filter_error:
                logger.warning(f"Filtering failed: {filter_error}")
                # Use all candidates if filtering fails
                filtered = candidates_dict
            
            # Format results
            formatted = self._format_results_from_dict(filtered)
            
            # YELLOW LINE 4 - FIX: Added email sending with comprehensive error handling
            email_sent = False
            try:
                send_email_results(formatted, self.memory.job_memory.job_title)
                logger.info("✓ Results sent via email")
                email_sent = True
            except Exception as email_error:
                logger.warning(f"Email sending failed: {email_error}")
                logger.info("Results still available in memory, but not emailed")
            
            return {
                "query_processed": True,
                "candidates_returned": len(filtered),
                "email_sent": email_sent
            }
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
    
    def _evaluations_to_dict(self, 
                            evaluations: List[CandidateEvaluation]) -> List[Dict]:
        """
        Convert CandidateEvaluation objects to dictionary format.
        
        Args:
            evaluations: List of CandidateEvaluation objects
        
        Returns:
            List of candidate dictionaries
        """
        return [
            {
                "name": c.candidate_name,
                "score": c.scores.final_score,
                "skills": c.skills,
                "experience": c.experience,
                "strengths": c.strengths,
                "weaknesses": c.weaknesses,
                "job_titles": c.job_titles,
                "resume_id": c.resume_id
            }
            for c in evaluations
        ]
    
    def _validate_evaluation_data(self, evaluation_data: Dict) -> None:
        """
        Validate LLM response has required keys and valid values.
        
        Args:
            evaluation_data: Dictionary from LLM
        
        Raises:
            ValueError: If required keys missing or invalid
        """
        required_keys = [
            "skill_score", 
            "project_score", 
            "clarity_score", 
            "strengths", 
            "weaknesses"
        ]
        
        # Check all required keys present
        for key in required_keys:
            if key not in evaluation_data:
                raise ValueError(f"Missing required key in LLM response: {key}")
        
        # Validate scores are floats in 0-10 range
        for score_key in ["skill_score", "project_score", "clarity_score"]:
            try:
                score = float(evaluation_data[score_key])
                if not 0 <= score <= 10:
                    raise ValueError(f"{score_key} must be 0-10, got {score}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"{score_key} validation failed: {e}")
        
        # Validate lists
        if not isinstance(evaluation_data.get("strengths"), list):
            raise ValueError("strengths must be list")
        if not isinstance(evaluation_data.get("weaknesses"), list):
            raise ValueError("weaknesses must be list")
    
    def _get_default_evaluation(self) -> Dict:
        """
        Get default evaluation when LLM fails.
        
        Returns:
            Dictionary with default scores
        """
        return {
            "skill_score": 5.0,
            "project_score": 5.0,
            "clarity_score": 5.0,
            "strengths": ["Unable to evaluate"],
            "weaknesses": ["LLM evaluation failed"],
            "skills": [],
            "experience": "Unknown",
            "job_titles": []
        }
    
    def _format_results_from_dict(self, candidates: list) -> dict:
        """
        Format filtered candidates for email delivery.
        
        Args:
            candidates: List of candidate dictionaries
        
        Returns:
            Formatted results dictionary for email
        """
        return {
            "job_title": (
                self.memory.job_memory.job_title 
                if self.memory.job_memory else "N/A"
            ),
            "evaluated_at": datetime.now().isoformat(),
            "total_candidates": len(candidates),
            "candidates": [
                {
                    "rank": idx + 1,
                    "name": c["name"],
                    "score": f"{c['score']:.2f}",
                    "skills": c.get("skills", []),
                    "experience": c.get("experience", ""),
                    "job_titles": c.get("job_titles", []),
                    "strengths": c.get("strengths", []),
                    "weaknesses": c.get("weaknesses", []),
                }
                for idx, c in enumerate(candidates)
            ]
        }
    
    def _run_batch(self, jobs: List[Dict]) -> Dict:
        """
        Run batch mode: evaluate same resumes against multiple jobs.
        
        Args:
            jobs: List of job dictionaries [{id, title, description}, ...]
        
        Returns:
            Batch results dictionary with per-job results
        """
        logger.info(f"Batch mode: Processing {len(jobs)} jobs...")
        
        if not self.memory.has_resumes():
            logger.error("Cannot run batch mode without resumes loaded")
            return {
                "success": False,
                "error": "No resumes loaded",
                "mode": "batch",
                "total_jobs": len(jobs),
                "results": {}
            }
        
        batch_results = {}
        successful = 0
        failed = 0
        
        for idx, job in enumerate(jobs, start=1):
            job_id = job.get("id", f"job_{idx}")
            job_title = job.get("title", f"Job {idx}")
            job_description = job.get("description", "")
            
            logger.info(f"[{idx}/{len(jobs)}] Processing: {job_title}")
            
            # YELLOW LINE 5 - FIX: Better error handling for single job processing
            try:
                job_result = self._process_single_job(job_title, job_description)
                batch_results[job_id] = job_result
                successful += 1
            except Exception as e:
                logger.error(f"Failed to process {job_title}: {e}")
                batch_results[job_id] = {
                    "success": False,
                    "error": str(e),
                    "job_title": job_title
                }
                failed += 1
        
        logger.info(
            f"Batch complete. Success: {successful}/{len(jobs)}, "
            f"Failed: {failed}"
        )
        
        return {
            "success": successful > 0,
            "mode": "batch",
            "total_jobs": len(jobs),
            "successful": successful,
            "failed": failed,
            "results": batch_results
        }
    
    def _process_single_job(self, job_title: str, job_description: str) -> Dict:
        """
        Process a single job: JOB_SETUP → EVALUATION → FORMAT.
        
        Args:
            job_title: Job title
            job_description: Job description text
        
        Returns:
            Job result dictionary with ranked candidates
        """
        # Create temporary context for this job
        temp_context = StateContext()
        old_context = self.context
        self.context = temp_context
        
        self.context.job_description = job_description
        
        try:
            self.context.transition_to(AgentState.JOB_SETUP)
            self._handle_job_setup()
            
            self.context.transition_to(AgentState.EVALUATION)
            self._handle_evaluation()
            
            # Get ranked results
            ranked = self.memory.candidate_memory.rank_candidates()
            formatted = self._format_results(ranked)
            
            return {
                "success": True,
                "job_title": job_title,
                "candidates_evaluated": len(ranked),
                "top_candidate": {
                    "name": ranked[0].candidate_name,
                    "score": f"{ranked[0].scores.final_score:.2f}",
                    "rank": 1
                } if ranked else None,
                "ranked_candidates": formatted["candidates"]
            }
        
        finally:
            # Restore original context
            self.context = old_context
    
    def _format_results(self, evaluations: list) -> dict:
        """
        Format evaluation results for output.
        
        Args:
            evaluations: List of CandidateEvaluation objects
        
        Returns:
            Formatted results dictionary
        """
        return {
            "job_title": (
                self.memory.job_memory.job_title 
                if self.memory.job_memory else "N/A"
            ),
            "evaluated_at": datetime.now().isoformat(),
            "total_candidates": len(evaluations),
            "candidates": [
                {
                    "rank": eval.rank,
                    "name": eval.candidate_name,
                    "score": f"{eval.scores.final_score:.2f}",
                    "skills": eval.skills,
                    "experience": eval.experience,
                    "job_titles": eval.job_titles,
                    "strengths": eval.strengths,
                    "weaknesses": eval.weaknesses,
                }
                for eval in evaluations
            ]
        }
    
    def _get_result(self) -> dict:
        """
        Return final execution result.
        
        Returns:
            Final result dictionary with status and summary
        """
        return {
            "success": self.context.error is None,
            "error": self.context.error,
            "state": self.context.current_state.value,
            "summary": self.memory.get_evaluation_summary(),
        }