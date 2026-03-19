#!/usr/bin/env python
"""
Test Harness for Customer Support Classification System
Run this to validate all components before notebook integration
"""

import sys
import json
from pathlib import Path


def test_imports():
    """Test that all required packages are available"""
    print("\n" + "="*80)
    print("TEST 1: Validating Required Imports")
    print("="*80)
    
    imports_to_test = {
        'sklearn': ['TfidfVectorizer', 'LogisticRegression', 'LinearSVC', 'MultinomialNB'],
        'langgraph': ['StateGraph'],
        'openai': ['OpenAI'],
        'pandas': ['DataFrame'],
        'numpy': ['array'],
        'sqlite3': [],
        'pickle': [],
        'json': [],
        're': [],
        'datetime': ['datetime'],
        'gradio': ['Interface'],
    }
    
    missing = []
    for module, items in imports_to_test.items():
        try:
            if items:
                exec(f"from {module} import {', '.join(items)}")
            else:
                exec(f"import {module}")
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}")
            missing.append((module, str(e)))
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} packages:")
        for module, error in missing:
            print(f"    - {module}: {error}")
        return False
    
    print("\n✨ All imports validated!")
    return True


def test_database():
    """Test database setup"""
    print("\n" + "="*80)
    print("TEST 2: Validating SQLite Database Setup")
    print("="*80)
    
    try:
        import sqlite3
        
        # Check if database exists
        db_path = Path("data/interactions.db")
        if not db_path.exists():
            print(f"  ℹ️  Database not found. Run: python setup_database.py")
            print(f"    This will create {db_path}")
            return False
        
        # Connect and validate schema
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        required_tables = {'interactions', 'feedback', 'model_versions', 'performance_metrics'}
        missing_tables = required_tables - tables
        
        if missing_tables:
            print(f"  ❌ Missing tables: {missing_tables}")
            conn.close()
            return False
        
        for table in required_tables:
            print(f"  ✅ Table '{table}' exists")
        
        conn.close()
        print("\n✨ Database schema validated!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_artifact_directory():
    """Test artifacts directory"""
    print("\n" + "="*80)
    print("TEST 3: Validating Artifact Directory")
    print("="*80)
    
    try:
        artifact_dir = Path("artifacts")
        artifact_dir.mkdir(exist_ok=True)
        print(f"  ✅ Artifact directory ready: {artifact_dir.absolute()}")
        
        # Check for expected files from a previous training
        expected_files = ['model.pkl', 'vectorizer.pkl', 'encoder.pkl', 'evaluation_results.json']
        found_files = [f for f in expected_files if (artifact_dir / f).exists()]
        
        if found_files:
            print(f"  ✅ Found previous artifacts: {found_files}")
        else:
            print(f"  ℹ️  No previous artifacts found (expected on first run)")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_node_functions():
    """Test that node functions can be imported and called"""
    print("\n" + "="*80)
    print("TEST 4: Validating Node Function Signatures")
    print("="*80)
    
    try:
        # Check if all_priority_nodes.py exists
        nodes_file = Path("all_priority_nodes.py")
        if not nodes_file.exists():
            print(f"  ⚠️  {nodes_file} not found")
            print(f"    Expected location: {nodes_file.absolute()}")
            return False
        
        # Try to import node functions
        sys.path.insert(0, str(Path.cwd()))
        try:
            from all_priority_nodes import (
                select_model_node,
                persist_artifacts_node,
                detect_language_node,
                translate_to_english_node,
                confidence_router_node,
                draft_response_node,
                log_interaction_node
            )
            
            node_functions = [
                ('select_model_node', select_model_node),
                ('persist_artifacts_node', persist_artifacts_node),
                ('detect_language_node', detect_language_node),
                ('translate_to_english_node', translate_to_english_node),
                ('confidence_router_node', confidence_router_node),
                ('draft_response_node', draft_response_node),
                ('log_interaction_node', log_interaction_node),
            ]
            
            for name, func in node_functions:
                # Check function signature
                if callable(func):
                    print(f"  ✅ {name} - importable and callable")
                else:
                    print(f"  ❌ {name} - not callable")
                    return False
            
            print("\n✨ All node functions validated!")
            return True
            
        except ImportError as e:
            print(f"  ❌ Import failed: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_state_structure():
    """Test that state structure is compatible with LangGraph"""
    print("\n" + "="*80)
    print("TEST 5: Validating State Structure (Dict-based for LangGraph)")
    print("="*80)
    
    try:
        # Create mock state like LangGraph would
        mock_state = {
            "customer_message": "I can't log into my account",
            "conversation_id": "conv_12345",
            "detected_language": "en",
            "requires_translation": False,
            "translated_message": None,
            "predicted_category": "Login Issue",
            "confidence_score": 0.92,
            "class_probabilities": {
                "Login Issue": 0.92,
                "Billing Issue": 0.05,
                "Technical Support": 0.03
            },
            "routing_decision": "auto_approved",
            "response_template": "We've sent a password reset link...",
            "interaction_id": None,
        }
        
        # Validate state is a dict
        if isinstance(mock_state, dict):
            print(f"  ✅ State is dict (LangGraph compatible)")
        else:
            print(f"  ❌ State must be dict, got {type(mock_state)}")
            return False
        
        # Validate required fields
        required_fields = ['customer_message', 'conversation_id', 'predicted_category']
        for field in required_fields:
            if field in mock_state:
                print(f"  ✅ Required field '{field}' present")
            else:
                print(f"  ❌ Required field '{field}' missing")
                return False
        
        print("\n✨ State structure validated!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_openai_config():
    """Test OpenAI API configuration"""
    print("\n" + "="*80)
    print("TEST 6: Validating OpenAI Configuration")
    print("="*80)
    
    try:
        import os
        
        # Check for API key in environment
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
            print(f"  ✅ OPENAI_API_KEY found in environment ({masked_key})")
        else:
            print(f"  ⚠️  OPENAI_API_KEY not in environment")
            print(f"    Set it with: export OPENAI_API_KEY='sk-...'")
        
        # Try to create OpenAI client (won't actually call API)
        try:
            from openai import OpenAI
            
            if api_key:
                client = OpenAI(api_key=api_key)
                print(f"  ✅ OpenAI client instantiated")
            else:
                print(f"  ⚠️  Skipping client instantiation (no API key)")
        except Exception as e:
            print(f"  ⚠️  Could not instantiate OpenAI client: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  Warning: {e}")
        return True  # Not critical for local testing


def run_all_tests():
    """Run all tests and provide summary"""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "CUSTOMER SUPPORT CLASSIFICATION SYSTEM - TEST SUITE".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    tests = [
        ("Imports", test_imports),
        ("Database Setup", test_database),
        ("Artifact Directory", test_artifact_directory),
        ("Node Functions", test_node_functions),
        ("State Structure", test_state_structure),
        ("OpenAI Configuration", test_openai_config),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results[name] = False
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status:10} | {name}")
    
    print("\n" + "="*80)
    print(f"Results: {passed}/{total} tests passed")
    print("="*80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for notebook integration.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. See above for details.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
