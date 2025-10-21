import asyncio
import os
from pathlib import Path

from dataqa import CoreRequest, CoreResponse, LocalClient


async def main():
    """
    Main function to initialize the client, send a query, and print the results.
    """
    print("🚀 Initializing DataQA LocalClient...")

    # 1. Define the path to the agent's configuration file.
    #    The LocalClient will use this to build the entire agent and its dependencies.
    SCRIPT_DIR = Path(__file__).resolve().parent
    config_path = SCRIPT_DIR / "cwd_agent_gemini.yaml"

    # 2. Instantiate the client.
    client = LocalClient(config_path=str(config_path))

    # 3. Define the query and create a request object.
    #    The conversation_id is used to maintain state between turns if needed.
    query = "What is the total gross sales volume by MOP code for co_id 1001 for Q12025 for Visa?"
    request = CoreRequest(
        user_query=query,
        conversation_id="local_test_session_01",  # A unique ID for this conversation.
    )

    print(f"\n▶️  Sending query: '{query}'")

    # 4. Process the query and await the response.
    #    This single call orchestrates the entire agentic workflow.
    response: CoreResponse = await client.process_query(request)

    # 5. Print the structured results from the CoreResponse object.
    print("\n" + "=" * 20 + " AGENT RESPONSE " + "=" * 20)
    print("\n📝 Final Text Response:")
    print(response.text)

    if response.output_dataframes:
        print("\n📊 Output DataFrames:")
        for i, df in enumerate(response.output_dataframes):
            print(f"\n--- DataFrame {i + 1} ---")
            # Using to_markdown for clean console output
            print(df.to_markdown(index=False))

    if response.output_images:
        print(f"\n🖼️  Generated {len(response.output_images)} image(s).")
        # For this script, we'll save the images to an 'output' directory.
        output_dir = SCRIPT_DIR / "output"
        output_dir.mkdir(exist_ok=True)
        for i, img_bytes in enumerate(response.output_images):
            img_path = output_dir / f"output_image_{i + 1}.png"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            print(f"   - Saved image to: {img_path}")

    # The CoreResponse also includes detailed steps for debugging.
    print("\n" + "=" * 20 + " DEBUG INFO " + "=" * 20)
    print("\n⚙️ Agent Execution Steps:")
    for step in response.steps:
        print(f"\n--- {step.name} ---")
        print(step.content)

    print("\n✅ Script finished.")


if __name__ == "__main__":
    # Ensure necessary environment variables are set before running.
    # The LocalClient relies on these for its LLM components.
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")

    # Run the asynchronous main function.
    asyncio.run(main())
