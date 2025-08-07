#!/usr/bin/env python3
"""
Startup script for the LLM-Powered Intelligent Query-Retrieval System
Provides easy commands to run the system in different modes
"""

import os
import sys
import subprocess
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import google.generativeai
        import faiss
        import sentence_transformers
        import PyPDF2
        import docx
        logger.info("All dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False

def setup_environment():
    """Setup environment variables and create necessary directories"""
    try:
        # Create data directory for FAISS index
        os.makedirs("data", exist_ok=True)
        
        # Check for .env file
        if not os.path.exists(".env"):
            if os.path.exists(".env.example"):
                logger.info("Creating .env from .env.example")
                with open(".env.example", 'r') as src, open(".env", 'w') as dst:
                    dst.write(src.read())
            else:
                logger.warning("No .env.example found. Creating basic .env file")
                with open(".env", 'w') as f:
                    f.write("GEMINI_API_KEY=your_gemini_api_key_here\n")
                    f.write("DEBUG=False\n")
                    f.write("LOG_LEVEL=INFO\n")
        
        logger.info("Environment setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        return False

def run_server(host="0.0.0.0", port=8000, reload=False):
    """Run the FastAPI server"""
    try:
        logger.info(f"Starting server on {host}:{port}")
        logger.info("Press Ctrl+C to stop the server")
        
        cmd = [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", host,
            "--port", str(port)
        ]
        
        if reload:
            cmd.append("--reload")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error running server: {e}")

def run_tests():
    """Run the test suite"""
    try:
        logger.info("Running test suite...")
        subprocess.run([sys.executable, "test_system.py"])
    except Exception as e:
        logger.error(f"Error running tests: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="LLM-Powered Intelligent Query-Retrieval System"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("serve", help="Start the API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run the test suite")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup environment and dependencies")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        # Check dependencies and setup
        if not check_dependencies():
            sys.exit(1)
        
        if not setup_environment():
            sys.exit(1)
        
        # Run server
        run_server(args.host, args.port, args.reload)
        
    elif args.command == "test":
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Run tests
        run_tests()
        
    elif args.command == "setup":
        # Setup environment
        setup_environment()
        logger.info("Setup completed. Run 'python start.py serve' to start the server")
        
    else:
        # Default: show help and run server
        parser.print_help()
        print("\n" + "="*60)
        print("Starting server with default settings...")
        print("="*60)
        
        if not check_dependencies():
            sys.exit(1)
        
        if not setup_environment():
            sys.exit(1)
        
        run_server()

if __name__ == "__main__":
    main() 