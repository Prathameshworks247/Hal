#!/usr/bin/env python3
"""
Test script for Session FAISS API endpoints.
Tests the new session-based conversational RAG system.
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_new_session_query():
    """Test 1: Query without session_id (should create new session)"""
    print("\n" + "="*60)
    print("TEST 1: New Session Query (No session_id)")
    print("="*60)
    
    payload = {
        "query": "What materials are commonly used in aircraft construction?",
        "file_name": "default",
        "conversation_history": []
    }
    
    print(f"\n📤 Sending request:")
    print(json.dumps(payload, indent=2))
    
    response = requests.post(f"{BASE_URL}/user/rectify", json=payload)
    
    print(f"\n📥 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        session_id = data.get("session_id")
        print(f"✅ Success! Session ID: {session_id}")
        print(f"\n📝 Response snippet:")
        print(f"   Query: {data.get('query', 'N/A')[:60]}...")
        print(f"   Status: {data.get('status', 'N/A')}")
        if 'rectification' in data:
            rec = data['rectification'].get('ai_recommendation', '')
            print(f"   Answer: {rec[:100]}...")
        return session_id
    else:
        print(f"❌ Failed: {response.text}")
        return None


def test_follow_up_query(session_id):
    """Test 2: Follow-up query with session_id (should use conversation memory)"""
    print("\n" + "="*60)
    print("TEST 2: Follow-Up Query (With session_id and history)")
    print("="*60)
    
    payload = {
        "query": "What about corrosion prevention for those materials?",
        "file_name": "default",
        "session_id": session_id,
        "conversation_history": [
            {
                "role": "user",
                "content": "What materials are commonly used in aircraft construction?"
            },
            {
                "role": "assistant",
                "content": "Aircraft commonly use aluminum alloys, titanium, and composite materials."
            }
        ]
    }
    
    print(f"\n📤 Sending request with session: {session_id}")
    print(f"   History: {len(payload['conversation_history'])} messages")
    
    response = requests.post(f"{BASE_URL}/user/rectify", json=payload)
    
    print(f"\n📥 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Session ID: {data.get('session_id')}")
        if 'conversation_context' in data:
            ctx = data['conversation_context']
            print(f"   Context used: {ctx.get('context_used', 'N/A')}")
        if 'rectification' in data:
            rec = data['rectification'].get('ai_recommendation', '')
            print(f"   Answer: {rec[:100]}...")
        return True
    else:
        print(f"❌ Failed: {response.text}")
        return False


def test_list_sessions():
    """Test 3: List all active sessions"""
    print("\n" + "="*60)
    print("TEST 3: List Active Sessions")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/admin/sessions")
    
    print(f"\n📥 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success!")
        print(f"   Active sessions: {data.get('active_sessions', 0)}")
        print(f"   Total storage: {data.get('total_storage_mb', 0)} MB")
        
        sessions = data.get('sessions', [])
        if sessions:
            print(f"\n   Sessions:")
            for s in sessions[:3]:  # Show first 3
                print(f"     - {s['session_id'][:16]}... ({s['conversation_turns']} turns)")
        return True
    else:
        print(f"❌ Failed: {response.text}")
        return False


def test_end_session(session_id):
    """Test 4: Delete a session"""
    print("\n" + "="*60)
    print("TEST 4: End Session (Delete)")
    print("="*60)
    
    print(f"\n🗑️  Deleting session: {session_id}")
    
    response = requests.delete(f"{BASE_URL}/user/end-session/{session_id}")
    
    print(f"\n📥 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success!")
        print(f"   Status: {data.get('status')}")
        print(f"   Deleted embeddings: {data.get('deleted_embeddings', 0)}")
        print(f"   Conversation turns: {data.get('conversation_turns', 0)}")
        return True
    else:
        print(f"❌ Failed: {response.text}")
        return False


def test_health_check():
    """Test 0: Check if server is running"""
    print("\n" + "="*60)
    print("TEST 0: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/system/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            return True
        else:
            print(f"⚠️  Server responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        print(f"   Expected URL: {BASE_URL}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    print("\n" + "🚀"*30)
    print("SESSION FAISS API TEST SUITE")
    print("🚀"*30)
    
    # Test 0: Health check
    if not test_health_check():
        print("\n❌ Server is not running. Please start it first:")
        print("   cd /Users/prathameshpatil/Sky-Sentinal")
        print("   source hall/bin/activate")
        print("   python app.py")
        return
    
    time.sleep(1)
    
    # Test 1: New session query
    session_id = test_new_session_query()
    if not session_id:
        print("\n❌ Test 1 failed. Stopping tests.")
        return
    
    time.sleep(2)
    
    # Test 2: Follow-up query with context
    if not test_follow_up_query(session_id):
        print("\n❌ Test 2 failed.")
    
    time.sleep(2)
    
    # Test 3: List sessions
    test_list_sessions()
    
    time.sleep(2)
    
    # Test 4: Delete session
    if session_id:
        test_end_session(session_id)
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)
    print("\n💡 TIP: Check the FastAPI logs to see session creation/deletion")


if __name__ == "__main__":
    main()

