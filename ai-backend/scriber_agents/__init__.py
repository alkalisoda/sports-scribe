"""AI Agents Package.

This package contains the various AI agents that make up the Sport Scribe content generation system:
- Data Collector Agent: Gathers game data from sports APIs
- Research Agent: Provides contextual background and analysis
- Writing Agent: Generates engaging sports articles
- Editor Agent: Reviews and refines article quality
- Format Manager: Handles article formatting and styling
- Article Pipeline: Orchestrates the complete article generation workflow
"""

from .data_collector import DataCollectorAgent
from .researcher import ResearchAgent
from .writer import WritingAgent
from .editor import EditorAgent
from .format_manager import FormatManager
from .pipeline import ArticlePipeline

__all__ = [
    "DataCollectorAgent",
    "ResearchAgent", 
    "WritingAgent",
    "EditorAgent",
    "FormatManager",
    "ArticlePipeline"
]
