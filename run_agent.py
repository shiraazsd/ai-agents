# --------------------------------------------------------------------------------------
# CLI ENTRYPOINT
# --------------------------------------------------------------------------------------
# USAGE EXAMPLES:
#   python run_agent.py "Explain the architecture" --stream
#   python run_agent.py "Show pdf references about topic X"
#   python run_agent.py "shell ls -1"   (tool route)
#
# FLAGS:
#   --stream : Enable token streaming (prints incrementally).
#
# IMPLEMENTATION:
#   - For streaming, we run an async loop and write tokens as received.
#   - For non-streaming, we simply invoke and print final answer.
#
# EXTENSIONS:
#   - Add --rollback <ckpt_id>, --history, --time-travel <n>.
#   - Add JSON output mode for programmatic integration.
# --------------------------------------------------------------------------------------
import asyncio, argparse, sys
from agent.graph import run_agent, run_agent_stream

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="User input")
    ap.add_argument("--stream", action="store_true", help="Stream answer tokens")
    args = ap.parse_args()

    if args.stream:
        async def _s():
            async for token in run_agent_stream(args.query):
                sys.stdout.write(str(token))
                sys.stdout.flush()
            print()
        asyncio.run(_s())
    else:
        result = run_agent(args.query)
        print(result.get("answer","(no answer)"))

if __name__ == "__main__":
    main()