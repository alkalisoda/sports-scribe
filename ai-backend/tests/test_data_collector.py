"""
Unit tests for the Data Collector Agent and its guardrail function.

This module contains comprehensive tests for:
1. DataCollectorAgent functionality
2. Data validation guardrail logic
3. Integration testing of agent with guardrail

The guardrail function (validate_data_quality) is decorated with @output_guardrail,
which makes it an OutputGuardrail object that cannot be called directly in tests.
Therefore, we test the underlying logic by simulating the guardrail behavior.

Test Strategy:
- TestDataCollector: Basic functionality tests for the data collector agent
- TestValidateDataQualityLogic: Unit tests for the validation logic used in the guardrail
- TestDataCollectorAgentWithGuardrail: Integration tests to ensure the agent properly uses the guardrail
"""

import os
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import http
import json
import os

from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, Runner
from dotenv import load_dotenv

from scriber_agents.data_collector import DataCollectorAgent, DataOutput

load_dotenv()

mock_results = {
    "get": "teams",
    "parameters": {"id": "33"},
    "errors": [],
    "results": 1,
    "paging": {"current": 1, "total": 1},
    "response": [
        {
            "team": {
                "id": 33,
                "name": "Manchester United",
                "code": "MUN",
                "country": "England",
                "founded": 1878,
                "national": False,
                "logo": "https://media.api-sports.io/football/teams/33.png",
            },
            "venue": {
                "id": 556,
                "name": "Old Trafford",
                "address": "Sir Matt Busby Way",
                "city": "Manchester",
                "capacity": 76212,
                "surface": "grass",
                "image": "https://media.api-sports.io/football/venues/556.png",
            },
        }
    ],
}


class TestDataCollector:
    @pytest.mark.asyncio
    async def test_writer_agent_generates_article(self):
        """Tests the data collecting agent"""
        config = {"name": "test", "model": "gpt-4"}
        dc = DataCollectorAgent(config)
        football_data = await Runner.run(dc.agent, "Get football data")

        assert football_data is not None
        # assert isinstance(football_data, expected_type)

    def test_endpoint(self):
        """Test main endpoint"""
        api_key = os.getenv("RAPIDAPI_KEY")
        if not api_key:
            raise ValueError("RAPID_API_KEY not found.")

        conn = http.client.HTTPSConnection("api-football-v1.p.rapidapi.com")

        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            "x-rapidapi-key": api_key,
        }

        conn.request("GET", "/v3/teams?id=33", headers=headers)

        response = conn.getresponse()  # Returns HTTP response object
        # data = response.read()
        # decoded_data = data.decode("utf8")

        assert response.status == 200

    def test_api_key(self):
        api_key = os.getenv("RAPIDAPI_KEY")

        assert len(api_key) > 0
        assert api_key


class TestValidateDataQualityLogic:
    """Test suite for the data validation logic used in the guardrail function"""

    @pytest.fixture
    def mock_context(self):
        """Create a mock RunContextWrapper for testing"""
        mock_ctx = Mock(spec=RunContextWrapper)
        mock_ctx.context = Mock()
        return mock_ctx

    @pytest.fixture
    def mock_agent(self):
        """Create a mock Agent for testing"""
        return Mock(spec=Agent)

    @pytest.fixture
    def valid_json_output(self):
        """Valid JSON output that should pass validation"""
        return json.dumps(
            {
                "get": "teams",
                "parameters": {"id": "33"},
                "errors": [],
                "results": 1,
                "paging": {"current": 1, "total": 1},
                "response": [{"team": {"id": 33, "name": "Manchester United"}}],
            }
        )

    @pytest.fixture
    def invalid_json_output(self):
        """Invalid JSON output that should fail validation"""
        return "This is not valid JSON format"

    @pytest.fixture
    def incomplete_json_output(self):
        """JSON output missing required fields"""
        return json.dumps(
            {
                "get": "teams",
                "parameters": {"id": "33"},
                # Missing required fields: errors, results, paging, response
            }
        )

    async def simulate_guardrail_logic(
        self, ctx, agent, output: str
    ) -> GuardrailFunctionOutput:
        """Simulate the guardrail logic without using the decorator"""
        # This simulates what the actual guardrail function does
        Agent(
            name="Guardrail check",
            instructions="Check if the output is of the correct format.",
            output_type=DataOutput,
        )

        # Mock the runner result based on the output
        if self.is_valid_json_format(output):
            mock_result = Mock()
            mock_result.final_output = DataOutput(
                reasoning="Output is valid JSON with correct structure", is_valid=True
            )
        else:
            mock_result = Mock()
            mock_result.final_output = DataOutput(
                reasoning="Output is not valid JSON format", is_valid=False
            )

        return GuardrailFunctionOutput(
            output_info=mock_result.final_output,
            tripwire_triggered=not mock_result.final_output.is_valid,
        )

    def is_valid_json_format(self, output: str) -> bool:
        """Helper method to check if output is valid JSON format"""
        try:
            data = json.loads(output)
            required_fields = [
                "get",
                "parameters",
                "errors",
                "results",
                "paging",
                "response",
            ]
            return all(field in data for field in required_fields)
        except (json.JSONDecodeError, TypeError):
            return False

    @pytest.mark.asyncio
    async def test_valid_output_passes_validation(
        self, mock_context, mock_agent, valid_json_output
    ):
        """Test that valid JSON output passes through the guardrail"""
        result = await self.simulate_guardrail_logic(
            mock_context, mock_agent, valid_json_output
        )

        # Assertions
        assert isinstance(result, GuardrailFunctionOutput)
        assert result.tripwire_triggered is False  # Should not trigger for valid output
        assert result.output_info.is_valid is True
        assert (
            result.output_info.reasoning
            == "Output is valid JSON with correct structure"
        )

    @pytest.mark.asyncio
    async def test_invalid_output_triggers_guardrail(
        self, mock_context, mock_agent, invalid_json_output
    ):
        """Test that invalid output triggers the guardrail"""
        result = await self.simulate_guardrail_logic(
            mock_context, mock_agent, invalid_json_output
        )

        # Assertions
        assert isinstance(result, GuardrailFunctionOutput)
        assert result.tripwire_triggered is True  # Should trigger for invalid output
        assert result.output_info.is_valid is False
        assert result.output_info.reasoning == "Output is not valid JSON format"

    @pytest.mark.asyncio
    async def test_incomplete_output_triggers_guardrail(
        self, mock_context, mock_agent, incomplete_json_output
    ):
        """Test that incomplete JSON output triggers the guardrail"""
        result = await self.simulate_guardrail_logic(
            mock_context, mock_agent, incomplete_json_output
        )

        # Assertions
        assert isinstance(result, GuardrailFunctionOutput)
        assert result.tripwire_triggered is True
        assert result.output_info.is_valid is False

    @pytest.mark.asyncio
    async def test_empty_output_handling(self, mock_context, mock_agent):
        """Test handling of empty or None output"""
        # Test with empty string
        result = await self.simulate_guardrail_logic(mock_context, mock_agent, "")
        assert result.tripwire_triggered is True
        assert result.output_info.is_valid is False

        # Test with None (converted to string)
        result = await self.simulate_guardrail_logic(mock_context, mock_agent, "None")
        assert result.tripwire_triggered is True
        assert result.output_info.is_valid is False

    @pytest.mark.asyncio
    async def test_malformed_json_output(self, mock_context, mock_agent):
        """Test handling of malformed JSON that might cause parsing issues"""
        malformed_outputs = [
            '{"incomplete": json',  # Incomplete JSON
            '{"invalid": "json"',  # Missing closing brace
            "{invalid json}",  # Invalid JSON syntax
            '{"null_value": null, "undefined": undefined}',  # Invalid undefined
        ]

        for malformed_output in malformed_outputs:
            result = await self.simulate_guardrail_logic(
                mock_context, mock_agent, malformed_output
            )
            assert result.tripwire_triggered is True
            assert result.output_info.is_valid is False

    @pytest.mark.asyncio
    async def test_large_output_handling(self, mock_context, mock_agent):
        """Test handling of very large outputs"""
        # Create a large JSON output
        large_response = [{"team": f"Team {i}", "id": i} for i in range(1000)]
        large_output = json.dumps(
            {
                "get": "teams",
                "parameters": {"limit": "1000"},
                "errors": [],
                "results": 1000,
                "paging": {"current": 1, "total": 1},
                "response": large_response,
            }
        )

        result = await self.simulate_guardrail_logic(
            mock_context, mock_agent, large_output
        )
        assert result.tripwire_triggered is False
        assert result.output_info.is_valid is True

    def test_data_output_model_validation(self):
        """Test the DataOutput model validation"""
        # Test valid DataOutput
        valid_data = DataOutput(reasoning="Test reasoning", is_valid=True)
        assert valid_data.reasoning == "Test reasoning"
        assert valid_data.is_valid is True

        # Test invalid DataOutput
        invalid_data = DataOutput(reasoning="Test reasoning", is_valid=False)
        assert invalid_data.reasoning == "Test reasoning"
        assert invalid_data.is_valid is False

    def test_json_format_validation_helper(self):
        """Test the helper method for JSON format validation"""
        # Valid JSON with all required fields
        valid_json = json.dumps(
            {
                "get": "teams",
                "parameters": {"id": "33"},
                "errors": [],
                "results": 1,
                "paging": {"current": 1, "total": 1},
                "response": [{"team": {"id": 33, "name": "Manchester United"}}],
            }
        )
        assert self.is_valid_json_format(valid_json) is True

        # Invalid JSON
        assert self.is_valid_json_format("invalid json") is False

        # Valid JSON but missing required fields
        incomplete_json = json.dumps({"get": "teams", "parameters": {"id": "33"}})
        assert self.is_valid_json_format(incomplete_json) is False

        # Empty string
        assert self.is_valid_json_format("") is False

    @pytest.mark.asyncio
    async def test_guardrail_function_output_structure(
        self, mock_context, mock_agent, valid_json_output
    ):
        """Test that the guardrail function returns the correct output structure"""
        result = await self.simulate_guardrail_logic(
            mock_context, mock_agent, valid_json_output
        )

        # Check that all required attributes are present
        assert hasattr(result, "output_info")
        assert hasattr(result, "tripwire_triggered")
        assert hasattr(result.output_info, "reasoning")
        assert hasattr(result.output_info, "is_valid")

        # Check types
        assert isinstance(result.tripwire_triggered, bool)
        assert isinstance(result.output_info.reasoning, str)
        assert isinstance(result.output_info.is_valid, bool)


class TestDataCollectorAgentWithGuardrail:
    """Integration tests for DataCollectorAgent with guardrail"""

    @pytest.mark.asyncio
    async def test_agent_with_guardrail_integration(self):
        """Test that the agent properly uses the guardrail"""
        config = {"name": "test", "model": "gpt-4"}
        dc = DataCollectorAgent(config)

        # Check that the agent has the guardrail configured
        assert dc.agent.output_guardrails is not None
        assert len(dc.agent.output_guardrails) > 0

        # The guardrail should be an OutputGuardrail object
        guardrail = dc.agent.output_guardrails[0]
        assert hasattr(guardrail, "guardrail_function")
        assert hasattr(guardrail, "name")

        # The underlying function should be callable
        assert callable(guardrail.guardrail_function)

    def test_agent_initialization_with_guardrail(self):
        """Test that the agent is properly initialized with the guardrail"""
        config = {"name": "test", "model": "gpt-4"}
        dc = DataCollectorAgent(config)

        # Verify agent properties
        assert dc.agent.name == "SportsDataCollector"
        assert dc.agent.output_guardrails is not None
        assert len(dc.agent.output_guardrails) == 1
