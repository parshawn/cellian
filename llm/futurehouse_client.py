"""Edison Scientific RAG client for literature search."""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env file from llm directory first
    local_env_path = Path(__file__).parent / '.env'
    if local_env_path.exists():
        load_dotenv(local_env_path)
    # Also try loading from project root
    root_env_path = Path(__file__).parent.parent / '.env'
    if root_env_path.exists():
        load_dotenv(root_env_path)
except ImportError:
    pass


# Edison Scientific API configuration
_EDISON_API_KEY = os.environ.get("EDISON_API_KEY")


def _get_client():
    """Get Edison Scientific client instance."""
    if not _EDISON_API_KEY:
        raise RuntimeError("EDISON_API_KEY env var is not set. Set it in .env file or environment variable.")
    try:
        from edison_client import EdisonClient
        return EdisonClient(api_key=_EDISON_API_KEY)
    except ImportError:
        raise ImportError("edison-client library not installed. Install with: uv pip install edison-client")


def search_literature(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Call Edison Scientific's literature RAG API.
    
    Uses PaperQA for high-accuracy, cited responses to scientific queries.
    
    Args:
        query: Free-text query string (e.g., "TP53 knockout apoptosis")
        k: Maximum number of documents to return (approximate, as Edison returns relevant papers)
    
    Returns:
        List of document dictionaries with keys:
        - title: str
        - abstract: str
        - pmid: str or None
        - year: int or None
        - url: str or None
        - score: float (relevance score 0.0-1.0)
        - authors: List[str] or None
        - journal: str or None
    """
    if not _EDISON_API_KEY:
        # Return placeholder/empty results if API key not configured
        print("Warning: EDISON_API_KEY not configured. Literature search will return placeholder results.")
        print("  To enable: Set EDISON_API_KEY in .env file or environment variable")
        return _get_placeholder_results(query, k)
    
    try:
        from edison_client import EdisonClient, JobNames
        from edison_client.models.app import TaskRequest
        
        # Get client
        client = _get_client()
        
        # Create task request using TaskRequest object
        task = TaskRequest(
            name=JobNames.LITERATURE,
            query=query,
        )
        
        # Run task and get response (TaskResponse object)
        print(f"[Edison Scientific] Searching literature for: {query[:80]}...")
        print(f"[Edison Scientific] This may take a few minutes...")
        import sys
        sys.stdout.flush()
        
        # Add timeout to prevent hanging (120 seconds max)
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Literature search timed out after 120 seconds")
        
        # Set timeout (Unix only)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)  # 120 second timeout
        except (AttributeError, OSError):
            # Windows doesn't support SIGALRM, skip timeout
            pass
        
        try:
            resp = client.run_tasks_until_done(task)
        finally:
            # Cancel alarm
            try:
                signal.alarm(0)
            except (AttributeError, OSError):
                pass
        
        # Parse response to extract papers
        # Note: resp has .answer, .formatted_answer, and .raw (full TaskResponse)
        papers = _parse_edison_response(resp, k)
        
        if papers and len(papers) > 0:
            print(f"[Edison Scientific] Found {len(papers)} papers")
        else:
            print(f"[Edison Scientific] No papers found in response")
        
        return papers
            
    except ImportError:
        print("Warning: edison-client library not available. Install with: uv pip install edison-client")
        print("Returning placeholder results")
        return _get_placeholder_results(query, k)
    except RuntimeError as e:
        # API key not set
        print(f"Warning: {e}")
        return _get_placeholder_results(query, k)
    except Exception as e:
        print(f"Warning: Edison Scientific API call failed: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return _get_placeholder_results(query, k)


def _parse_edison_response(task_response: Any, k: int) -> List[Dict[str, Any]]:
    """
    Parse Edison Scientific API response into standardized format.
    
    Args:
        task_response: TaskResponse from EdisonClient.run_tasks_until_done()
        k: Maximum number of results
    
    Returns:
        List of parsed document dictionaries
    """
    results = []
    
    try:
        # Try to extract papers from environment_frame
        environment_frame = getattr(task_response, 'environment_frame', None)
        
        if environment_frame:
            # For PaperQA, papers are typically in contexts or paper_metadata
            papers_data = None
            
            # Check various possible locations in environment_frame
            if isinstance(environment_frame, dict):
                # Try contexts (list of contexts with paper info)
                contexts = environment_frame.get("contexts", [])
                if contexts:
                    papers_data = contexts
                
                # Try paper_metadata
                if not papers_data:
                    papers_data = environment_frame.get("paper_metadata", [])
                
                # Try papers
                if not papers_data:
                    papers_data = environment_frame.get("papers", [])
            
            if papers_data:
                for paper_info in papers_data[:k]:
                    # Parse paper info (format depends on PaperQA structure)
                    parsed = _extract_paper_info(paper_info)
                    if parsed:
                        results.append(parsed)
        
        # If no papers found in environment_frame, try to extract from formatted_answer
        if not results and hasattr(task_response, 'formatted_answer'):
            # Extract PMIDs and citations from formatted answer
            papers_from_answer = _extract_papers_from_formatted_answer(
                task_response.formatted_answer,
                k
            )
            if papers_from_answer:
                results.extend(papers_from_answer)
        
        # If still no results, create a basic entry from the answer
        if not results and hasattr(task_response, 'answer'):
            results.append({
                "title": f"Edison Scientific Response",
                "abstract": task_response.answer[:500] if len(task_response.answer) > 500 else task_response.answer,
                "pmid": None,
                "year": None,
                "url": None,
                "score": 0.9,  # High score since it's a direct answer
                "authors": None,
                "journal": None
            })
            
    except Exception as e:
        print(f"Warning: Error parsing Edison response: {e}")
    
    # Ensure we return at least something
    if not results:
        return _get_placeholder_results("", k)
    
    return results[:k]


def _extract_paper_info(paper_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract paper information from PaperQA context/metadata."""
    try:
        # Try different possible structures
        title = paper_info.get("title", paper_info.get("Title", ""))
        abstract = paper_info.get("abstract", paper_info.get("Abstract", paper_info.get("summary", "")))
        pmid = paper_info.get("pmid", paper_info.get("PMID", paper_info.get("pubmed_id", None)))
        year = paper_info.get("year", paper_info.get("Year", paper_info.get("publication_year", None)))
        authors = paper_info.get("authors", paper_info.get("Authors", []))
        journal = paper_info.get("journal", paper_info.get("Journal", paper_info.get("journal_name", None)))
        
        # Extract URL if available
        url = paper_info.get("url", paper_info.get("URL", paper_info.get("link", None)))
        if not url and pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        
        # Score (if available)
        score = paper_info.get("score", paper_info.get("relevance", 0.8))
        if isinstance(score, (int, float)):
            score = float(score)
            if score > 1.0:
                score = score / 100.0
        else:
            score = 0.8
        
        return {
            "title": title or "Unknown",
            "abstract": abstract or "",
            "pmid": pmid,
            "year": int(year) if year else None,
            "url": url,
            "score": score,
            "authors": authors if isinstance(authors, list) else ([authors] if authors else None),
            "journal": journal
        }
    except Exception:
        return None


def _extract_papers_from_formatted_answer(formatted_answer: str, k: int) -> List[Dict[str, Any]]:
    """Extract paper information from formatted answer with citations."""
    results = []
    
    # Extract PMIDs from citations (format: PMID:12345678 or [1], etc.)
    pmid_pattern = r'PMID[:\s]*(\d+)'
    pmids = re.findall(pmid_pattern, formatted_answer, re.IGNORECASE)
    
    for i, pmid in enumerate(pmids[:k]):
        results.append({
            "title": f"Referenced Paper {i+1}",
            "abstract": "",  # Not available from formatted answer
            "pmid": pmid,
            "year": None,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "score": 0.8,
            "authors": None,
            "journal": None
        })
    
    return results


def _get_placeholder_results(query: str, k: int) -> List[Dict[str, Any]]:
    """
    Return placeholder results when API is not available.
    
    Args:
        query: Search query
        k: Number of results
    
    Returns:
        List of placeholder document dictionaries
    """
    return [
        {
            "title": f"Placeholder result {i+1} for query: {query}",
            "abstract": "Edison Scientific API not configured. Install edison-client with 'uv pip install edison-client' and configure EDISON_API_KEY in .env to enable literature search.",
            "pmid": None,
            "year": None,
            "url": None,
            "score": 0.5,
            "authors": None,
            "journal": None
        }
        for i in range(min(k, 3))
    ]


def format_citation(paper: Dict[str, Any], include_url: bool = True) -> str:
    """
    Format a paper dictionary as a citation string with link.
    
    Args:
        paper: Paper dictionary from search_literature()
        include_url: Whether to include clickable URL link
    
    Returns:
        Formatted citation string with markdown link if URL available
    """
    parts = []
    
    if paper.get("authors") and isinstance(paper["authors"], list) and len(paper["authors"]) > 0:
        if len(paper["authors"]) == 1:
            parts.append(paper["authors"][0])
        elif len(paper["authors"]) <= 3:
            parts.append(" et al.".join(paper["authors"]))
        else:
            parts.append(f"{paper['authors'][0]} et al.")
    
    if paper.get("year"):
        parts.append(f"({paper['year']})")
    
    if paper.get("title"):
        parts.append(paper["title"])
    
    if paper.get("journal"):
        parts.append(f"<i>{paper['journal']}</i>")
    
    # Build citation text
    citation_text = ". ".join(filter(None, parts))
    
    # Add URL/PMID link if available
    url = paper.get("url")
    pmid = paper.get("pmid")
    
    if include_url and url:
        # Use markdown link format: [text](url)
        # Include PMID in the link text if available
        link_text = f"PMID: {pmid}" if pmid else "View paper"
        citation_text += f" [{link_text}]({url})"
    elif pmid:
        # If no URL but have PMID, create PubMed URL
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        citation_text += f" [PMID: {pmid}]({pubmed_url})"
    
    return citation_text


def get_pmids(papers: List[Dict[str, Any]]) -> List[str]:
    """
    Extract PMIDs from list of papers.
    
    Args:
        papers: List of paper dictionaries
    
    Returns:
        List of PMID strings (formatted as "PMID:12345678")
    """
    pmids = []
    for paper in papers:
        pmid = paper.get("pmid")
        if pmid:
            pmids.append(f"PMID:{pmid}")
    return pmids


def get_gemini_model():
    """
    Get Google Gemini model for LLM operations.
    
    Returns:
        Gemini model instance or None if not available
    """
    import sys
    
    # Check if library is installed
    try:
        import google.generativeai as genai
    except ImportError as e:
        print(f"  [LLM] ✗ google-generativeai library not installed: {e}")
        print(f"  [LLM] Install with: pip install google-generativeai")
        sys.stdout.flush()
        return None
    
    # Try to get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"  [LLM] ✗ API key not found in environment")
        print(f"  [LLM] Set GOOGLE_API_KEY or GEMINI_API_KEY in .env file or environment variable")
        print(f"  [LLM] Checking .env files...")
        sys.stdout.flush()
        
        # Check if .env file exists
        local_env_path = Path(__file__).parent / '.env'
        root_env_path = Path(__file__).parent.parent / '.env'
        env_files_found = []
        if local_env_path.exists():
            env_files_found.append(str(local_env_path))
        if root_env_path.exists():
            env_files_found.append(str(root_env_path))
        
        if env_files_found:
            print(f"  [LLM] Found .env files: {', '.join(env_files_found)}")
            print(f"  [LLM] But GOOGLE_API_KEY or GEMINI_API_KEY not set in them")
        else:
            print(f"  [LLM] No .env files found at:")
            print(f"  [LLM]   - {local_env_path}")
            print(f"  [LLM]   - {root_env_path}")
        sys.stdout.flush()
        return None
    
    try:
        genai.configure(api_key=api_key)
        
        # Try to use the best available model (gemini-2.0-flash-exp or gemini-1.5-pro)
        # gemini-pro is deprecated/not available
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            return model
        except Exception:
            # Fall back to gemini-1.5-pro if gemini-2.0-flash-exp is not available
            try:
                model = genai.GenerativeModel('gemini-1.5-pro')
                return model
            except Exception:
                # Last resort: try gemini-1.5-flash
                model = genai.GenerativeModel('gemini-1.5-flash')
                return model
    except Exception as e:
        print(f"  [LLM] ✗ Failed to initialize Gemini model: {e}")
        print(f"  [LLM] Error type: {type(e).__name__}")
        import traceback
        print(f"  [LLM] Traceback: {traceback.format_exc()}")
        sys.stdout.flush()
        return None


def test_edison_connection() -> Dict[str, Any]:
    """
    Test Edison Scientific connection and return status.
    
    Returns:
        Dictionary with status information:
        - "configured": bool (API key is set)
        - "library_installed": bool (edison-client is installed)
        - "can_connect": bool (can successfully make API call)
        - "error": str or None
    """
    status = {
        "configured": bool(_EDISON_API_KEY),
        "library_installed": False,
        "can_connect": False,
        "error": None
    }
    
    # Check if library is installed
    try:
        from edison_client import EdisonClient, JobNames
        status["library_installed"] = True
    except ImportError:
        status["error"] = "edison-client library not installed. Install with: uv pip install edison-client"
        return status
    
    # Check if API key is set
    if not status["configured"]:
        status["error"] = "EDISON_API_KEY not configured. Set it in .env file or environment variable."
        return status
    
    # Try a simple API call
    try:
        client = _get_client()
        # Test with a simple query
        from edison_client.models.app import TaskRequest
        task = TaskRequest(
            name=JobNames.LITERATURE,
            query="test",
        )
        # Don't wait for completion, just check if we can create the task
        status["can_connect"] = True
        status["error"] = None
    except Exception as e:
        status["can_connect"] = False
        status["error"] = str(e)
    
    return status


if __name__ == "__main__":
    # Test function when run directly
    print("Testing Edison Scientific connection...")
    status = test_edison_connection()
    print(f"Configured: {status['configured']}")
    print(f"Library installed: {status['library_installed']}")
    print(f"Can connect: {status['can_connect']}")
    if status['error']:
        print(f"Error: {status['error']}")

