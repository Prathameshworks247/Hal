#!/usr/bin/env python3
"""
Test script to demonstrate intent classification in Sky-Sentinel system.
This script sends test queries to demonstrate SNAG, INSPECTION, and CONCEPTUAL intents.
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:8000/rectify"

# Test queries for each intent type
test_queries = {
    "SNAG": [
        "Engine oil pressure low during flight",
        "Hydraulic leak detected in landing gear",
        "APU fails to start after maintenance",
        "Navigation system showing intermittent errors"
    ],
    "INSPECTION": [
        "How to inspect the main rotor blades?",
        "What are the steps for pre-flight inspection?",
        "Daily inspection checklist for hydraulic system",
        "Describe the 100-hour inspection procedures"
    ],
    "CONCEPTUAL": [
        "How does the hydraulic system work?",
        "What is the purpose of the pitot tube?",
        "Explain the principle of autorotation",
        "What are the components of the fuel system?"
    ]
}


def test_intent_classification(query, intent_type, file_name="default", pb_number="TEST"):
    """
    Test a single query and print the response.
    
    Args:
        query: The query to test
        intent_type: Expected intent type (SNAG, INSPECTION, or CONCEPTUAL)
        file_name: File name for the query (default: "default")
        pb_number: PB number (default: "TEST")
    """
    print(f"\n{'='*80}")
    print(f"TESTING {intent_type} INTENT")
    print(f"{'='*80}")
    print(f"Query: {query}")
    print(f"-"*80)
    
    payload = {
        "query": query,
        "file_name": file_name,
        "pb_number": pb_number
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            # Pretty print the response
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print("✅ Response received:")
                print(json.dumps(result, indent=2))
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to the API.")
        print("   Make sure the server is running: uvicorn app:app --reload")
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Request took too long.")
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")


def run_all_tests():
    """Run all test queries for all intent types."""
    print("\n" + "="*80)
    print("SKY-SENTINEL INTENT CLASSIFICATION TEST SUITE")
    print("="*80)
    
    for intent_type, queries in test_queries.items():
        for query in queries:
            test_intent_classification(query, intent_type)
            print("\n" + "."*80)
            input("Press Enter to continue to next test...")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)


def interactive_mode():
    """Interactive mode to test custom queries."""
    print("\n" + "="*80)
    print("SKY-SENTINEL INTENT CLASSIFICATION - INTERACTIVE MODE")
    print("="*80)
    print("\nEnter your queries to test the intent classification.")
    print("The system will automatically determine if it's a SNAG, INSPECTION, or CONCEPTUAL query.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        query = input("\nEnter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nExiting interactive mode. Goodbye!")
            break
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        test_intent_classification(query, "AUTO-DETECTED")


def main():
    """Main entry point."""
    print("\nSky-Sentinel Intent Classification Test")
    print("\nChoose a test mode:")
    print("1. Run all predefined tests")
    print("2. Interactive mode (enter custom queries)")
    print("3. Quick single test")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        confirm = input("\nThis will run multiple tests. Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            run_all_tests()
        else:
            print("Test cancelled.")
    elif choice == "2":
        interactive_mode()
    elif choice == "3":
        print("\nQuick test examples:")
        print("1. SNAG: 'Engine oil pressure low'")
        print("2. INSPECTION: 'How to inspect landing gear?'")
        print("3. CONCEPTUAL: 'How does the hydraulic system work?'")
        
        query = input("\nEnter your query: ").strip()
        if query:
            test_intent_classification(query, "QUICK-TEST")
        else:
            print("No query entered.")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    # Check if server is running first
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            main()
        else:
            print("⚠️ Server is running but health check failed.")
            proceed = input("Continue anyway? (y/n): ").strip().lower()
            if proceed == 'y':
                main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to the server.")
        print("\nPlease start the server first:")
        print("   cd /Users/prathameshpatil/Sky-Sentinal")
        print("   source hall/bin/activate")
        print("   uvicorn app:app --reload")
        print("\nThen run this test script again.")
    except Exception as e:
        print(f"\n❌ Unexpected error checking server status: {e}")

