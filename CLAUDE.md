# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### AI Backend (Python)
```bash
cd ai-backend
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python main.py  # Start FastAPI server on port 8000
```

### Web Platform (Next.js)
```bash
cd web
npm install
npm run dev  # Start Next.js dev server on port 3000
npm run build  # Production build
npm run lint  # ESLint
```

### Testing
```bash
# Run all tests
./scripts/run-tests.sh

# Individual components
./scripts/run-tests.sh ai      # AI backend tests (pytest)
./scripts/run-tests.sh web     # Web platform tests
./scripts/run-tests.sh lint    # Linting only
```

### Code Quality
```bash
# Comprehensive linting and quality checks
./scripts/lint-all.sh

# Auto-fix linting issues  
./scripts/lint-fix.sh [ai|web|sql|all]

# Type checking
./scripts/type-check.sh [ai|web|all]
```

### Docker Development
```bash
# Start both services
docker-compose -f docker-compose.dev.yml up

# Individual services
docker-compose -f docker-compose.dev.yml up ai-backend
docker-compose -f docker-compose.dev.yml up web
```

### Database Management
```bash
cd web
npm run db:setup           # Reset and seed database
npm run generate:types     # Generate TypeScript types from Supabase
```

## Project Architecture

### Multi-Agent AI System
The AI backend uses a pipeline architecture with specialized agents:

- **DataCollectorAgent** (`scriber_agents/data_collector.py`): Fetches sports data from APIs
- **ResearchAgent** (`scriber_agents/researcher.py`): Analyzes team/player backgrounds  
- **WriterAgent** (`scriber_agents/writer.py`): Generates articles with specified tone/style
- **Editor** (`scriber_agents/editor.py`): Reviews and improves content quality
- **AgentPipeline** (`scriber_agents/pipeline.py`): Orchestrates the workflow

Pipeline flow: Data Collector → Researcher → Writer → Editor

### Backend Structure
- `main.py`: FastAPI application entry point with article generation endpoints
- `config/`: Agent configurations and application settings
- `tools/`: Sports APIs (`sports_apis.py`), data validation, web search utilities  
- `utils/`: Logging, security, helper functions
- `tests/`: Pytest test suite with agent and API tests

### Frontend Structure  
- Next.js 14 with App Router and TypeScript
- **HeroUI** (@heroui/react) component library, not standard Material-UI or Chakra
- `app/`: App router pages including admin dashboard and article views
- `components/`: Reusable React components organized by feature
- `lib/`: Supabase client, utilities, AI integration, webhook handlers
- `hooks/`: Custom React hooks for data fetching

### Shared Resources
- `shared/types/`: TypeScript interfaces for articles, games, players, teams
- `shared/schemas/`: Database SQL schemas and JSON validation schemas  
- `shared/constants/`: API endpoints, leagues, sports data

## Configuration Files

### Python (AI Backend)
- `ruff.toml`: Python linting with strict rules, Google docstring convention
- `mypy.ini`: Type checking configuration
- `pytest.ini`: Test configuration with async support
- `requirements.txt`: Production dependencies including security fixes for CVE vulnerabilities

### TypeScript (Web)
- `next.config.js`: Next.js configuration
- `tailwind.config.js`: Tailwind CSS setup
- `tsconfig.json`: TypeScript compiler options

## Environment Setup

Required environment variables:
- `OPENAI_API_KEY`: OpenAI API access
- `RAPIDAPI_KEY`: Sports data APIs
- `NEXT_PUBLIC_SUPABASE_PROJECT_ID`: Supabase project
- `SUPABASE_SERVICE_ROLE_KEY`: Database access

See `env.example` files in root, `ai-backend/`, and `web/` directories.

## Development Notes

- The AI system is currently basic/foundational with room for expansion
- Always activate Python virtual environment before backend development
- Use HeroUI components, not other UI libraries
- Database uses Supabase (PostgreSQL) with real-time capabilities
- Security: Fixed CVE vulnerabilities in Python dependencies
- Code quality enforced via ruff (Python) and ESLint (TypeScript)