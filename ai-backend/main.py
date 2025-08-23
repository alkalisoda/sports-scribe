"""Sport Scribe AI Backend - Main Application.

This is the main entry point for the AI backend system that orchestrates
the multi-agent sports journalism workflow.
"""

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from sciber_agents.data_collector import DataCollectorAgent
from sciber_agents.editor import EditorAgent
from sciber_agents.researcher import ResearchAgent
from sciber_agents.writer import WritingAgent

from config.agent_config import AgentConfigurations
from config.settings import get_settings
from utils.logging import get_logger, setup_logging

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Get application settings
settings = get_settings()


class ArticleRequest(BaseModel):
    """Request model for article generation."""

    game_id: str
    article_type: str = "game_recap"
    target_length: int = 800
    priority: str = "normal"
    tone: str = "professional"


class ArticleResponse(BaseModel):
    """Response model for article generation."""

    article_id: str
    status: str
    content: str = ""
    metadata: dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str = "1.0.0"
    environment: str
    agents_status: dict[str, str]


class AgentOrchestrator:
    """Orchestrates the multi-agent workflow for sports article generation."""

    def __init__(self) -> None:
        """Initialize the agent orchestrator with all agents."""
        logger.info("Initializing Agent Orchestrator")

        # Get agent configurations
        configs = AgentConfigurations.get_all_configs()

        # Initialize agents
        self.data_collector = DataCollectorAgent(configs["data_collector"].parameters)
        self.researcher = ResearchAgent(configs["researcher"].parameters)
        self.writer = WritingAgent(configs["writer"].parameters)
        self.editor = EditorAgent(configs["editor"].parameters)

        logger.info("All agents initialized successfully")

    async def generate_article(self, request: ArticleRequest) -> ArticleResponse:
        """Generate article using AI agents."""
        try:
            # Validate request
            if not request.game_id:
                raise HTTPException(status_code=400, detail="Game ID is required")

            # Initialize agents with default configurations
            configs = AgentConfigurations.get_all_configs()
            data_collector = DataCollectorAgent(configs["data_collector"].parameters)
            researcher = ResearchAgent(configs["researcher"].parameters)
            writer = WritingAgent(configs["writer"].parameters)
            editor = EditorAgent(configs["editor"].parameters)

            # Collect game data
            game_data = await self._collect_game_data(data_collector, request.game_id)

            # Research background information
            research_data = await self._research_background(researcher, game_data)

            # Generate article content
            article_content = await self._generate_content(
                writer, game_data, research_data, request
            )
            # Edit and finalize
            final_article = await self._edit_content(editor, article_content)

            return ArticleResponse(
                article_id=str(uuid.uuid4()),
                status="completed",
                content=final_article.get("content", ""),
                metadata=final_article.get("metadata", {}),
            )

        except Exception as e:
            logger.error(f"Error generating article: {e!s}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate article: {e!s}"
            ) from e

    async def _collect_game_data(
        self, collector: DataCollectorAgent, game_id: str
    ) -> dict[str, Any]:
        """Collect game data using data collector agent."""
        return await collector.collect_game_data(game_id)

    async def _research_background(
        self, researcher: ResearchAgent, game_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Research background information."""
        home_team = game_data.get("home_team", "")
        away_team = game_data.get("away_team", "")
        return await researcher.research_team_history(home_team, away_team)

    async def _generate_content(
        self,
        writer: WritingAgent,
        game_data: dict[str, Any],
        research_data: dict[str, Any],
        request: ArticleRequest,
    ) -> dict[str, Any]:
        """Generate article content."""
        content = await writer.generate_game_recap(game_data, research_data)
        return {
            "content": content,
            "metadata": {
                "tone": request.tone,
                "target_length": request.target_length,
                "article_type": request.article_type,
            },
        }

    async def _edit_content(
        self, editor: EditorAgent, content: dict[str, Any]
    ) -> dict[str, Any]:
        """Edit and finalize content."""
        article_content = content.get("content", "")
        metadata = content.get("metadata", {})
        edited_content, review_feedback = await editor.review_article(
            article_content, metadata
        )
        return {
            "content": edited_content,
            "metadata": {**metadata, "review_feedback": review_feedback},
        }


# Global orchestrator instance
orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    global orchestrator

    # Startup
    logger.info(
        "Starting Sport Scribe AI Backend",
        environment=settings.environment,
        debug=settings.debug,
        log_level=settings.log_level,
        version="1.0.0",
    )

    try:
        orchestrator = AgentOrchestrator()
        logger.info("Agent orchestrator initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize agent orchestrator", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("Shutting down Sport Scribe AI Backend")
    orchestrator = None


# Create FastAPI application
app = FastAPI(
    title="Sport Scribe AI Backend",
    description="Multi-agent AI system for generating sports articles",
    version="1.0.0",
    lifespan=lifespan,
)

# Add security middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add trusted host middleware for security
allowed_hosts = (
    ["*"]
    if settings.environment == "development"
    else [
        "localhost",
        "127.0.0.1",
        "sport-scribe.vercel.app",
        # Add other trusted hosts as needed
    ]
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",  # Alternative localhost
        "https://sport-scribe.vercel.app",  # Production domain
        # Add other trusted origins as needed
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        environment=settings.environment,
        agents_status={
            "data_collector": "ready",
            "researcher": "ready",
            "writer": "ready",
            "editor": "ready",
        },
    )


@app.post("/generate-article", response_model=ArticleResponse)
async def generate_article(
    request: ArticleRequest, background_tasks: BackgroundTasks
) -> ArticleResponse:
    """Generate a sports article using the multi-agent system."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not ready")

    return await orchestrator.generate_article(request)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Sport Scribe AI Backend",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting server on %s:%s", settings.fastapi_host, settings.fastapi_port
    )
    uvicorn.run(
        "main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=settings.fastapi_reload,
        log_level=settings.log_level.lower(),
    )
