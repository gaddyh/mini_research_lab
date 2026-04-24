#!/usr/bin/env python3
"""Debug chat interface to see what's happening with family detection."""

from chat import ExperimentTools

def main():
    tools = ExperimentTools()
    data = tools.list_available_data()
    
    print("🔍 Debug: Available Data")
    print("=" * 50)
    
    for symbol, families in data["symbols"].items():
        print(f"\n🎯 {symbol}:")
        for family in families:
            print(f"  - '{family}'")
    
    print(f"\n📊 All families: {data['families']}")
    
    # Test specific match
    print("\n🔍 Testing specific match:")
    question = "summary for AAPL ma_distance_reversion"
    question_lower = question.lower()
    
    for symbol in data["symbols"]:
        if symbol.lower() in question_lower:
            print(f"✅ Found symbol: {symbol}")
            for family in data["symbols"][symbol]:
                if family.lower() in question_lower:
                    print(f"✅ Found family: {family}")
                    summary = tools.get_summary(symbol, family)
                    print(f"📄 Summary keys: {list(summary.keys())}")
                    return
    
    print("❌ No match found")

if __name__ == "__main__":
    main()
