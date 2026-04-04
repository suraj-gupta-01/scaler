#!/usr/bin/env python3
"""
Test script to verify endpoints are accessible.
Usage: python test_endpoints.py https://tusharp2006-scaler_deployment.hf.space
"""

import sys
import httpx

async def test_endpoints(base_url: str):
    """Test all key endpoints"""
    endpoints = [
        "/",
        "/health",
        "/metrics",
        "/tasks",
    ]
    
    base_url = base_url.rstrip("/")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"
            try:
                resp = await client.get(url)
                print(f"[{resp.status_code}] GET {endpoint}")
                if resp.status_code < 300:
                    print(f"  ✓ Response: {str(resp.json())[:100]}")
                else:
                    print(f"  ✗ Error: {resp.text[:100]}")
            except Exception as e:
                print(f"[ERROR] GET {endpoint} - {e}")

if __name__ == "__main__":
    import asyncio
    if len(sys.argv) < 2:
        print("Usage: python test_endpoints.py <base_url>")
        print("Example: python test_endpoints.py https://tusharp2006-scaler_deployment.hf.space")
        sys.exit(1)
    
    asyncio.run(test_endpoints(sys.argv[1]))
