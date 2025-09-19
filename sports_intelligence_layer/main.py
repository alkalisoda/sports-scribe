"""
Main entry point for the Soccer Intelligence Layer.
Demonstrates the complete end-to-end flow: Query → Parse → SQL → Results
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from src.query_parser import SoccerQueryParser, ParsedSoccerQuery
from src.database import SoccerDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SoccerIntelligenceLayer:
    """
    Main class that orchestrates the complete end-to-end flow:
    Query → Parse → SQL → Results
    """

    def __init__(
        self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None
    ):
        """
        Initialize the Soccer Intelligence Layer.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key
        """
        # Load environment variables
        load_dotenv()

        # Get Supabase credentials
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase credentials not found. Please set SUPABASE_URL and "
                "SUPABASE_SERVICE_ROLE_KEY environment variables or pass them directly."
            )

        # Initialize components
        self.parser = SoccerQueryParser()
        self.database = SoccerDatabase(self.supabase_url, self.supabase_key)

        logger.info("Soccer Intelligence Layer initialized successfully")

    async def close(self) -> None:
        """
        Close all connections and clean up resources.

        This should be called before application exit to ensure:
        - All database connections are properly closed
        - Cache connections are flushed and closed
        - Resources are freed
        """
        try:
            await self.database.close()
            logger.info("✅ Soccer Intelligence Layer cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with automatic cleanup."""
        await self.close()

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language soccer query through the complete pipeline.

        Args:
            query: Natural language query (e.g., "How many goals has Haaland scored this season?")

        Returns:
            Dictionary containing the complete result with metadata
        """
        logger.info(f"=== PROCESSING QUERY: '{query}' ===")

        try:
            # Step 1: Parse the query
            logger.info("Step 1: Parsing query...")
            parsed_query = self.parser.parse_query(query)
            logger.info(
                f"✓ Query parsed successfully. Confidence: {parsed_query.confidence:.2f}"
            )

            # Step 2: Execute the query against the database
            logger.info("Step 2: Executing database query...")
            result = await self.database.run_from_parsed(parsed_query)
            logger.info("✓ Database query executed successfully")

            # Step 3: Format the response
            logger.info("Step 3: Formatting response...")
            response = self._format_response(query, parsed_query, result)
            logger.info("✓ Response formatted successfully")

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "status": "error",
                "message": str(e),
                "query": query,
                "timestamp": self._get_timestamp(),
            }

    def _format_response(
        self,
        original_query: str,
        parsed_query: ParsedSoccerQuery,
        db_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Format the final response with all relevant information.
        """
        response = {
            "status": "success",
            "query": {
                "original": original_query,
                "parsed": {
                    "entities": [
                        {
                            "name": entity.name,
                            "type": entity.entity_type.value,
                            "confidence": entity.confidence,
                        }
                        for entity in parsed_query.entities
                    ],
                    "time_context": parsed_query.time_context.value,
                    "statistic_requested": parsed_query.statistic_requested,
                    "comparison_type": (
                        parsed_query.comparison_type.value
                        if parsed_query.comparison_type
                        else None
                    ),
                    "filters": parsed_query.filters,
                    "intent": parsed_query.query_intent,
                    "confidence": parsed_query.confidence,
                },
            },
            "result": db_result,
            "metadata": {
                "timestamp": self._get_timestamp(),
                "processing_time_ms": 0,  # Could be calculated if needed
                "data_source": "supabase",
            },
        }

        return response

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.utcnow().isoformat()

    async def test_end_to_end(self) -> List[Dict[str, Any]]:
        """
        Run a comprehensive test of the end-to-end pipeline.
        """
        logger.info("=== RUNNING END-TO-END TESTS ===")

        test_queries = [
            "How many goals has Kaoru Mitoma scored this season?",
            "What's Danny Welbeck's assist record?",
            "How many minutes has Jordan Pickford played?",
            "Show me Dominic Calvert-Lewin's goals in the last 5 games",
            "What's João Pedro's performance at home?",
            "How many clean sheets has Jason Steele kept?",
        ]

        results = []
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n--- Test {i}/{len(test_queries)} ---")
            logger.info(f"Query: {query}")

            try:
                result = await self.process_query(query)
                results.append(
                    {
                        "test_number": i,
                        "query": query,
                        "status": result.get("status"),
                        "success": result.get("status") == "success",
                    }
                )

                if result.get("status") == "success":
                    logger.info("✓ Test passed")
                else:
                    logger.error(
                        f"✗ Test failed: {result.get('message', 'Unknown error')}"
                    )

            except Exception as e:
                logger.error(f"✗ Test failed with exception: {e}")
                results.append(
                    {
                        "test_number": i,
                        "query": query,
                        "status": "error",
                        "success": False,
                        "error": str(e),
                    }
                )

        # Summary
        successful_tests = sum(1 for r in results if r["success"])
        total_tests = len(results)

        logger.info("\n=== TEST SUMMARY ===")
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {total_tests - successful_tests}")
        logger.info(f"Success rate: {(successful_tests / total_tests) * 100:.1f}%")

        return results


async def main() -> None:
    """
    Main function to demonstrate the end-to-end functionality.

    Uses proper resource management with context managers to ensure
    all connections are properly closed before exit.
    """
    try:
        # Initialize the Soccer Intelligence Layer with proper cleanup
        logger.info("Initializing Soccer Intelligence Layer...")
        async with SoccerIntelligenceLayer() as sil:
            # Run end-to-end tests
            await sil.test_end_to_end()

            # Example of processing a single query
            logger.info("\n=== SINGLE QUERY EXAMPLE ===")
            example_query = "How many goals has Kaoru Mitoma scored this season?"
            result = await sil.process_query(example_query)

            logger.info(f"Query: {example_query}")
            logger.info(f"Result: {result}")

        # Context manager automatically calls close() here
        logger.info("✅ All resources cleaned up successfully")

    except Exception as e:
        logger.error(f"Failed to initialize or run tests: {e}")
        logger.error("Please ensure your environment variables are set correctly:")
        logger.error("- SUPABASE_URL")
        logger.error("- SUPABASE_SERVICE_ROLE_KEY")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
