"""Financial Optimization Orchestrator Agent.

This agent demonstrates the orchestrator-workers pattern using Claude Agent SDK.
It fetches financial data from MCP servers and coordinates specialized sub-agents
to provide comprehensive financial optimization recommendations.
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AgentDefinition,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    PermissionResultAllow,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


DATA_DIR: Path = Path(__file__).parent.parent / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw_data"
AGENT_OUTPUTS_DIR: Path = DATA_DIR / "agent_outputs"


def _ensure_directories():
    """Ensure all required directories exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(
    data: dict,
    filename: str
):
    """Save data to JSON file."""
    filepath = RAW_DATA_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved data to {filepath}")


def _load_prompt(filename: str) -> str:
    """Load prompt from prompts directory.

    Args:
        filename: Name of the prompt file

    Returns:
        Prompt text content
    """
    prompt_path = Path(__file__).parent / "prompts" / filename
    return prompt_path.read_text()


async def _auto_approve_all(
    tool_name: str,
    input_data: dict,
    context
):
    """Auto-approve all tools without prompting."""
    logger.debug(f"Auto-approving tool: {tool_name}")
    return PermissionResultAllow()


def _detect_subscriptions(
    bank_transactions: list[dict],
    credit_card_transactions: list[dict]
) -> list[dict]:
    """Detect subscription services from recurring transactions.

    Filters transactions marked as recurring and extracts subscription details.

    Args:
        bank_transactions: List of bank transaction dicts
        credit_card_transactions: List of credit card transaction dicts

    Returns:
        List of subscription dictionaries with service name, amount, frequency
    """
    subscriptions = []

    # Process bank transactions for recurring charges
    for transaction in bank_transactions:
        if transaction.get("recurring") and transaction.get("amount", 0) < 0:
            subscriptions.append({
                "service": transaction.get("description", "Unknown"),
                "amount": abs(transaction["amount"]),
                "frequency": "monthly",
            })

    # Process credit card transactions for recurring charges
    for transaction in credit_card_transactions:
        if transaction.get("recurring") and transaction.get("amount", 0) < 0:
            subscriptions.append({
                "service": transaction.get("merchant", "Unknown"),
                "amount": abs(transaction["amount"]),
                "frequency": "monthly",
            })

    return subscriptions


async def _fetch_financial_data(
    username: str,
    start_date: str,
    end_date: str
) -> tuple[dict, dict]:
    """Fetch data from Bank and Credit Card MCP servers.

    Uses a lightweight Claude agent with MCP server connections to call
    get_bank_transactions and get_credit_card_transactions tools,
    then saves the raw data to files.

    Args:
        username: Username for the account
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Tuple of (bank_data, credit_card_data) dictionaries
    """
    logger.info(f"Fetching financial data for {username} from {start_date} to {end_date}")

    # Configure MCP server connections
    mcp_servers = {
        "Bank Account Server": {
            "type": "http",
            "url": "http://127.0.0.1:5001/mcp"
        },
        "Credit Card Server": {
            "type": "http",
            "url": "http://127.0.0.1:5002/mcp"
        }
    }

    working_dir = Path(__file__).parent.parent  # personal-financial-analyst/

    # Use a simple agent to fetch data from MCP servers
    fetch_options = ClaudeAgentOptions(
        model="haiku",
        system_prompt="You are a data fetching assistant. Call the requested tools and save the results as JSON files. Do not ask questions, just execute.",
        mcp_servers=mcp_servers,
        can_use_tool=_auto_approve_all,
        cwd=str(working_dir),
    )

    bank_data = {}
    credit_card_data = {}

    try:
        async with ClaudeSDKClient(options=fetch_options) as client:
            await client.query(
                f"Please do the following:\n"
                f"1. Call get_bank_transactions with username='{username}', start_date='{start_date}', end_date='{end_date}'\n"
                f"2. Call get_credit_card_transactions with username='{username}', start_date='{start_date}', end_date='{end_date}'\n"
                f"3. Save the bank transaction results to data/raw_data/bank_transactions.json\n"
                f"4. Save the credit card transaction results to data/raw_data/credit_card_transactions.json\n"
                f"Execute both tool calls and save the results."
            )

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            logger.info(f"Fetch agent: {block.text[:100]}...")
                elif isinstance(message, ResultMessage):
                    logger.info(f"Data fetch complete. Duration: {message.duration_ms}ms, Cost: ${message.total_cost_usd:.4f}")
                    break

        # Read back the saved data
        bank_file = RAW_DATA_DIR / "bank_transactions.json"
        cc_file = RAW_DATA_DIR / "credit_card_transactions.json"

        if bank_file.exists():
            with open(bank_file, "r") as f:
                bank_data = json.load(f)
            logger.info(f"Loaded bank data: {len(bank_data.get('transactions', []))} transactions")

        if cc_file.exists():
            with open(cc_file, "r") as f:
                credit_card_data = json.load(f)
            logger.info(f"Loaded credit card data: {len(credit_card_data.get('transactions', []))} transactions")

    except Exception as e:
        logger.error(f"Error fetching financial data: {e}", exc_info=True)
        logger.error("Make sure MCP servers are running on ports 5001 and 5002")

    return bank_data, credit_card_data


async def _run_orchestrator(
    username: str,
    start_date: str,
    end_date: str,
    user_query: str
):
    """Main orchestrator agent logic.

    Implements the orchestrator-workers pattern:
    1. Fetches data from MCP servers
    2. Performs initial analysis (detect subscriptions)
    3. Defines and configures sub-agents
    4. Runs orchestrator with Claude Agent SDK
    5. Streams output and generates final report

    Args:
        username: Username for the account
        start_date: Start date for analysis
        end_date: End date for analysis
        user_query: User's financial question/request
    """
    logger.info("Starting financial optimization orchestrator")
    logger.info(f"User query: {user_query}")

    _ensure_directories()

    # Step 1: Fetch financial data from MCP servers
    bank_data, credit_card_data = await _fetch_financial_data(
        username,
        start_date,
        end_date
    )

    # Step 2: Initial analysis
    logger.info("Performing initial analysis...")

    bank_transactions = bank_data.get("transactions", [])
    credit_card_transactions = credit_card_data.get("transactions", [])

    subscriptions = _detect_subscriptions(
        bank_transactions,
        credit_card_transactions
    )

    logger.info(f"Detected {len(subscriptions)} subscriptions")

    # Step 3: Define sub-agents
    research_agent = AgentDefinition(
        description="Research cheaper alternatives for subscriptions and services",
        prompt=_load_prompt("research_agent_prompt.txt"),
        tools=["write"],
        model="haiku",
    )

    negotiation_agent = AgentDefinition(
        description="Create negotiation strategies and scripts for bills and services",
        prompt=_load_prompt("negotiation_agent_prompt.txt"),
        tools=["write"],
        model="haiku",
    )

    tax_agent = AgentDefinition(
        description="Identify tax-deductible expenses and optimization opportunities",
        prompt=_load_prompt("tax_agent_prompt.txt"),
        tools=["write"],
        model="haiku",
    )

    agents = {
        "research_agent": research_agent,
        "negotiation_agent": negotiation_agent,
        "tax_agent": tax_agent,
    }

    # Step 4: Configure orchestrator agent with sub-agents
    mcp_servers = {
        "Bank Account Server": {
            "type": "http",
            "url": "http://127.0.0.1:5001/mcp"
        },
        "Credit Card Server": {
            "type": "http",
            "url": "http://127.0.0.1:5002/mcp"
        }
    }

    working_dir = Path(__file__).parent.parent  # personal-financial-analyst/

    options = ClaudeAgentOptions(
        model="sonnet",
        system_prompt=_load_prompt("orchestrator_system_prompt.txt"),
        mcp_servers=mcp_servers,
        agents=agents,
        can_use_tool=_auto_approve_all,
        cwd=str(working_dir),
    )

    # Step 5: Run orchestrator with Claude Agent SDK
    # Build user prompt from template
    user_prompt_template = _load_prompt("orchestrator_user_prompt.txt")
    user_prompt = user_prompt_template.format(
        user_query=user_query,
        username=username,
        start_date=start_date,
        end_date=end_date,
    )

    # Add context about pre-fetched data and subscriptions
    subscription_summary = json.dumps(subscriptions, indent=2)
    prompt = f"""{user_prompt}

<pre_fetched_data>
The following data has already been fetched and saved to data/raw_data/:
- {len(bank_transactions)} bank transactions (saved to data/raw_data/bank_transactions.json)
- {len(credit_card_transactions)} credit card transactions (saved to data/raw_data/credit_card_transactions.json)
- {len(subscriptions)} identified subscriptions:
{subscription_summary}
</pre_fetched_data>

Please:
1. Identify opportunities for savings
2. Delegate research to the research agent
3. Delegate negotiation strategies to the negotiation agent
4. Delegate tax analysis to the tax agent
5. Read their results and create a final report at data/final_report.md
"""

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text, end='', flush=True)
                elif isinstance(message, ResultMessage):
                    logger.info(f"\nDuration: {message.duration_ms}ms")
                    logger.info(f"Cost: ${message.total_cost_usd:.4f}")
                    break

    except Exception as e:
        logger.error(f"Error during orchestration: {e}", exc_info=True)
        logger.error("\nTroubleshooting:")
        logger.error("1. Make sure MCP servers are running on ports 5001 and 5002")
        logger.error("2. Check that ANTHROPIC_API_KEY is set in .env")
        logger.error("3. If running inside Claude Code, unset CLAUDECODE first")
        raise

    # Step 6: Generate final report
    logger.info("Orchestration complete. Check data/final_report.md for results.")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Financial Optimization Orchestrator Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Basic analysis
    uv run python financial_orchestrator.py \\
        --username john_doe \\
        --start-date 2026-01-01 \\
        --end-date 2026-01-31 \\
        --query "How can I save $500 per month?"

    # Subscription analysis
    uv run python financial_orchestrator.py \\
        --username jane_smith \\
        --start-date 2026-01-01 \\
        --end-date 2026-01-31 \\
        --query "Analyze my subscriptions and find better deals"
"""
    )

    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Username for account (john_doe or jane_smith)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User's financial question or request"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = _parse_args()

    await _run_orchestrator(
        username=args.username,
        start_date=args.start_date,
        end_date=args.end_date,
        user_query=args.query
    )


if __name__ == "__main__":
    asyncio.run(main())
