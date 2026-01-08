#!/usr/bin/env python3
"""
Quick diagnostic script to check session status.
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def check_sessions():
    """Check all active sessions"""
    print("\n" + "="*60)
    print("🔍 CHECKING ACTIVE SESSIONS")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/admin/sessions")
        
        if response.status_code == 200:
            data = response.json()
            sessions = data.get('sessions', [])
            
            print(f"\n✅ Found {len(sessions)} active session(s)")
            print(f"📊 Total storage: {data.get('total_storage_mb', 0)} MB\n")
            
            if not sessions:
                print("⚠️  No active sessions found!")
                print("   This means no files have been uploaded yet.")
                return
            
            for i, session in enumerate(sessions, 1):
                print(f"\n{'─'*60}")
                print(f"Session #{i}")
                print(f"{'─'*60}")
                print(f"  ID: {session['session_id']}")
                print(f"  Created: {session['created_at']}")
                print(f"  Age: {session['age_hours']:.2f} hours")
                print(f"  Has uploaded file: {'✅ YES' if session['has_uploaded_file'] else '❌ NO'}")
                
                if session['has_uploaded_file']:
                    print(f"  File name: {session['uploaded_file_name']}")
                    print(f"  Embeddings: {session['total_embeddings']}")
                else:
                    print(f"  ⚠️  No file uploaded to this session!")
                
                print(f"  Conversation turns: {session['conversation_turns']}")
                
                # Diagnosis
                if not session['has_uploaded_file'] and session['conversation_turns'] == 0:
                    print(f"\n  💡 DIAGNOSIS: Empty session (no file, no conversations)")
                    print(f"     This session was created but never used.")
                elif not session['has_uploaded_file']:
                    print(f"\n  💡 DIAGNOSIS: Session has conversations but no uploaded file")
                    print(f"     Queries are using GLOBAL_FAISS only.")
                else:
                    print(f"\n  ✅ DIAGNOSIS: Healthy session with uploaded file")
        else:
            print(f"❌ Failed to get sessions: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server!")
        print(f"   Make sure server is running at {BASE_URL}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def check_specific_session(session_id):
    """Check a specific session"""
    print("\n" + "="*60)
    print(f"🔍 CHECKING SESSION: {session_id}")
    print("="*60)
    
    try:
        # Try to query with this session
        payload = {
            "query": "test query",
            "file_name": "test.pdf",
            "session_id": session_id,
            "conversation_history": []
        }
        
        response = requests.post(f"{BASE_URL}/user/rectify", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            if "error" in data:
                print(f"\n❌ Session returned error:")
                print(f"   {data['error']}")
                if 'suggestion' in data:
                    print(f"\n💡 Suggestion:")
                    print(f"   {data['suggestion']}")
            else:
                print(f"\n✅ Session is working!")
                print(f"   Status: {data.get('status')}")
        else:
            print(f"❌ Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Check specific session
        session_id = sys.argv[1]
        check_specific_session(session_id)
    else:
        # Check all sessions
        check_sessions()
    
    print("\n" + "="*60)
    print("💡 TIP: Run with session_id to check specific session:")
    print(f"   python check_sessions.py session_1767877674827")
    print("="*60 + "\n")

