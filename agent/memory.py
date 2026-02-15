"""Memory management for the agent."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Resume:
    """Single resume document."""
    
    resume_id: str
    candidate_name: str
    raw_text: str
    
    def __post_init__(self):
        """Validate resume data."""
        if not self.resume_id or not self.candidate_name or not self.raw_text:
            raise ValueError("resume_id, candidate_name, and raw_text are required")
        
        if not isinstance(self.resume_id, str):
            raise TypeError(f"resume_id must be string, got {type(self.resume_id)}")
        if not isinstance(self.candidate_name, str):
            raise TypeError(f"candidate_name must be string, got {type(self.candidate_name)}")
        if not isinstance(self.raw_text, str):
            raise TypeError(f"raw_text must be string, got {type(self.raw_text)}")


@dataclass
class EvaluationScores:
    """Sub-scores and final score for a candidate."""
    
    skill_score: float
    project_score: float
    clarity_score: float
    final_score: float = 0.0
    
    def __post_init__(self):
        """Validate scores are in 0-10 range."""
        for score_name, score_value in [
            ("skill_score", self.skill_score),
            ("project_score", self.project_score),
            ("clarity_score", self.clarity_score)
        ]:
            try:
                score_value = float(score_value)
                if not 0 <= score_value <= 10:
                    raise ValueError(f"{score_name} must be between 0 and 10, got {score_value}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"{score_name} validation failed: {e}")
    
    def compute_final_score(
        self,
        skill_weight: float = 0.5,
        project_weight: float = 0.3,
        clarity_weight: float = 0.2
    ) -> float:
        """
        Compute final score using weighted combination.
        
        Args:
            skill_weight: Weight for skill score (default 0.5)
            project_weight: Weight for project score (default 0.3)
            clarity_weight: Weight for clarity score (default 0.2)
        
        Returns:
            Final score between 0 and 10
        
        Raises:
            ValueError: If weights invalid
        """
        total_weight = skill_weight + project_weight + clarity_weight
        
        if total_weight <= 0:
            raise ValueError("Total weight must be greater than 0")
        
        self.final_score = (
            (self.skill_score * skill_weight) +
            (self.project_score * project_weight) +
            (self.clarity_score * clarity_weight)
        ) / total_weight
        
        return self.final_score


@dataclass
class CandidateEvaluation:
    """Evaluation result for a candidate."""
    
    resume_id: str
    candidate_name: str
    scores: EvaluationScores
    strengths: List[str]
    weaknesses: List[str]
    rank: Optional[int] = None
    skills: List[str] = field(default_factory=list)
    experience: str = ""
    job_titles: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate evaluation data."""
        if not self.resume_id or not self.candidate_name:
            raise ValueError("resume_id and candidate_name are required")
        if not isinstance(self.scores, EvaluationScores):
            raise TypeError("scores must be EvaluationScores instance")
        if not isinstance(self.strengths, list) or not isinstance(self.weaknesses, list):
            raise TypeError("strengths and weaknesses must be lists")


@dataclass
class JobMemory:
    """Details about the current job posting."""
    
    job_title: str
    job_description: str
    key_skills: List[str] = field(default_factory=list)
    key_requirements: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate job memory."""
        if not self.job_title or not self.job_description:
            raise ValueError("job_title and job_description are required")


class ResumeMemory:
    """Manages all loaded resumes."""
    
    def __init__(self):
        self._resumes: Dict[str, Resume] = {}
        logger.debug("ResumeMemory initialized")
    
    def add_resume(self, resume: Resume) -> None:
        """
        Add a resume.
        
        Args:
            resume: Resume object to add
        
        Raises:
            TypeError: If resume not Resume instance
        """
        if not isinstance(resume, Resume):
            raise TypeError("resume must be Resume instance")
        self._resumes[resume.resume_id] = resume
    
    def get_resume(self, resume_id: str) -> Optional[Resume]:
        """Get resume by ID."""
        return self._resumes.get(resume_id)
    
    def get_all_resumes(self) -> List[Resume]:
        """Get all resumes."""
        return list(self._resumes.values())
    
    def count(self) -> int:
        """Total number of resumes."""
        return len(self._resumes)
    
    def clear(self) -> None:
        """Clear all resumes."""
        self._resumes.clear()
        logger.debug("ResumeMemory cleared")


class CandidateMemory:
    """
    Manages evaluation results with proper ranking and caching.
    
    Features:
    - Efficient ranking with caching
    - Prevents duplicate rankings
    - Tracks ranking state
    """
    
    def __init__(self):
        self._evaluations: Dict[str, CandidateEvaluation] = {}
        self._ranked_list: List[str] = []
        self._is_ranked: bool = False  # Track if ranked (BUG FIX #4)
        logger.debug("CandidateMemory initialized")
    
    def add_evaluation(self, evaluation: CandidateEvaluation) -> None:
        """
        Add evaluation.
        
        Invalidates ranking when new evaluation added.
        
        Args:
            evaluation: CandidateEvaluation object
        
        Raises:
            TypeError: If evaluation not CandidateEvaluation instance
        """
        if not isinstance(evaluation, CandidateEvaluation):
            raise TypeError("evaluation must be CandidateEvaluation instance")
        
        self._evaluations[evaluation.resume_id] = evaluation
        
        # Adding new evaluation invalidates ranking (BUG FIX #4)
        self._is_ranked = False
    
    def get_evaluation(self, resume_id: str) -> Optional[CandidateEvaluation]:
        """Get evaluation by resume ID."""
        return self._evaluations.get(resume_id)
    
    def get_all_evaluations(self) -> List[CandidateEvaluation]:
        """Get all evaluations."""
        return list(self._evaluations.values())
    
    def rank_candidates(self) -> List[CandidateEvaluation]:
        """
        Sort candidates by score (highest first).
        
        Uses caching to avoid re-ranking if already ranked.
        Call clear_ranking() to force re-ranking.
        
        Returns:
            List of ranked CandidateEvaluation objects
        """
        # If already ranked, return cached result (BUG FIX #4)
        if self._is_ranked and self._ranked_list:
            logger.debug("Using cached ranking")
            return [
                self._evaluations[rid] 
                for rid in self._ranked_list 
                if rid in self._evaluations
            ]
        
        # Sort by score
        sorted_evals = sorted(
            self._evaluations.values(),
            key=lambda x: x.scores.final_score,
            reverse=True
        )
        
        # BUG FIX #1: Clear _ranked_list before adding new ranks
        self._ranked_list.clear()
        
        # Assign ranks and track order
        for rank, evaluation in enumerate(sorted_evals, start=1):
            evaluation.rank = rank
            self._ranked_list.append(evaluation.resume_id)
        
        # Mark as ranked
        self._is_ranked = True
        
        logger.info(f"Ranked {len(sorted_evals)} candidates")
        return sorted_evals
    
    def get_top_k(self, k: int) -> List[CandidateEvaluation]:
        """
        Get top K candidates.
        
        Uses pre-ranked list if available for efficiency.
        Otherwise sorts dynamically.
        
        Args:
            k: Number of top candidates to return
        
        Returns:
            List of top K candidates
        
        Raises:
            ValueError: If k invalid
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")
        
        # BUG FIX #5: Use pre-ranked list if available
        if self._is_ranked and self._ranked_list:
            logger.debug(f"Getting top {k} from pre-ranked list")
            return [
                self._evaluations[rid] 
                for rid in self._ranked_list[:k]
                if rid in self._evaluations
            ]
        
        # Fallback: sort if not ranked yet
        logger.debug(f"Re-sorting for top {k}")
        sorted_evals = sorted(
            self._evaluations.values(),
            key=lambda x: x.scores.final_score,
            reverse=True
        )
        return sorted_evals[:k]
    
    def get_ranked_ids(self) -> List[str]:
        """
        Get ranked candidate IDs in order.
        
        BUG FIX #3: This method now actually provides value!
        
        Returns:
            List of ranked resume IDs (highest to lowest score)
        
        Raises:
            ValueError: If not ranked yet
        """
        if not self._is_ranked or not self._ranked_list:
            raise ValueError("Candidates not ranked yet. Call rank_candidates() first.")
        
        return self._ranked_list.copy()
    
    def is_ranked(self) -> bool:
        """
        Check if candidates have been ranked.
        
        Returns:
            True if rank_candidates() was called and no new evaluations added
        """
        return self._is_ranked
    
    def clear_ranking(self) -> None:
        """
        Clear ranking to force re-ranking.
        
        Use when scores change and need new ranking.
        Also clears rank from all evaluation objects.
        """
        self._ranked_list.clear()
        self._is_ranked = False
        
        # Clear rank from all evaluations (BUG FIX #2: prevents stale ranks)
        for evaluation in self._evaluations.values():
            evaluation.rank = None
        
        logger.debug("Ranking cleared")
    
    def count(self) -> int:
        """Total number of evaluations."""
        return len(self._evaluations)
    
    def clear(self) -> None:
        """Clear all evaluations and ranking."""
        self._evaluations.clear()
        self._ranked_list.clear()
        self._is_ranked = False
        logger.debug("CandidateMemory cleared")


class AgentMemory:
    """Master memory manager for the agent."""
    
    def __init__(self):
        self.resume_memory = ResumeMemory()
        self.job_memory: Optional[JobMemory] = None
        self.candidate_memory = CandidateMemory()
        logger.debug("AgentMemory initialized")
    
    def start_new_job_session(self, job_memory: JobMemory) -> None:
        """
        Start new job evaluation session.
        
        Args:
            job_memory: JobMemory object
        
        Raises:
            TypeError: If job_memory not JobMemory instance
        """
        if not isinstance(job_memory, JobMemory):
            raise TypeError("job_memory must be JobMemory instance")
        self.job_memory = job_memory
        self.candidate_memory.clear()
        logger.info(f"Started new job session: {job_memory.job_title}")
    
    def has_resumes(self) -> bool:
        """Check if resumes loaded."""
        return self.resume_memory.count() > 0
    
    def has_job(self) -> bool:
        """Check if job loaded."""
        return self.job_memory is not None
    
    def get_evaluation_summary(self) -> Dict:
        """
        Get current state summary.
        
        Returns:
            Dictionary with summary stats
        """
        return {
            "resumes_loaded": self.resume_memory.count(),
            "job_title": self.job_memory.job_title if self.job_memory else None,
            "candidates_evaluated": self.candidate_memory.count(),
        }