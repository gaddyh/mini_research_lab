"""
User configuration system for experiment parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class UserConfig:
    """User configuration for experiment parameters."""
    symbols: List[str] = field(default_factory=lambda: ["AAPL"])
    start_date: str = "2015-01-01"
    end_date: str = "today"
    train_end_date: str = "2020-12-31"
    families: List[str] = field(default_factory=lambda: ["mean_reversion"])
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "UserConfig":
        """Create UserConfig from dictionary."""
        return cls(
            symbols=config_dict.get("symbols", ["AAPL"]),
            start_date=config_dict.get("start_date", "2015-01-01"),
            end_date=config_dict.get("end_date", "today"),
            train_end_date=config_dict.get("train_end_date", "2020-12-31"),
            families=config_dict.get("families", ["mean_reversion"])
        )
    
    @classmethod
    def from_json_file(cls, file_path: str) -> "UserConfig":
        """Load UserConfig from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert UserConfig to dictionary."""
        return {
            "symbols": self.symbols,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "train_end_date": self.train_end_date,
            "families": self.families
        }
    
    def to_json_file(self, file_path: str) -> None:
        """Save UserConfig to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_actual_end_date(self) -> str:
        """Get actual end date, converting 'today' to current date."""
        if self.end_date.lower() == "today":
            return datetime.now().strftime("%Y-%m-%d")
        return self.end_date
    
    def get_actual_train_end_date(self) -> str:
        """Get actual train end date, converting 'today' to current date."""
        if self.train_end_date.lower() == "today":
            return datetime.now().strftime("%Y-%m-%d")
        return self.train_end_date
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if not self.symbols:
            issues.append("No symbols provided")
        
        if not self.families:
            issues.append("No experiment families provided")
        
        valid_families = ["mean_reversion", "momentum", "volatility_clustering", "ma_distance_reversion", "rsi_mean_reversion", "rsi_bucket_analysis", "donchian_breakout_5d", "donchian_breakout_10d", "donchian_breakout_20d", "rsi_mean_reversion_event", "donchian_breakout_event", "ma_crossover_event"]
        for family in self.families:
            if family not in valid_families:
                issues.append(f"Invalid family: {family}. Valid options: {valid_families}")
        
        # Basic date format validation
        date_fields = ["start_date", "end_date", "train_end_date"]
        for field in date_fields:
            date_value = getattr(self, field)
            if date_value.lower() != "today":
                try:
                    datetime.strptime(date_value, "%Y-%m-%d")
                except ValueError:
                    issues.append(f"Invalid date format for {field}: {date_value}. Use YYYY-MM-DD format")
        
        return issues
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        print("📋 USER CONFIGURATION SUMMARY")
        print("=" * 50)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Start Date: {self.start_date}")
        print(f"End Date: {self.end_date}")
        print(f"Train End Date: {self.train_end_date}")
        print(f"Experiment Families: {', '.join(self.families)}")
        print("=" * 50)


def create_default_config() -> UserConfig:
    """Create default user configuration."""
    return UserConfig(
        symbols=["AAPL", "MSFT", "SPY"],
        start_date="2015-01-01",
        end_date="today",
        train_end_date="2020-12-31",
        families=["mean_reversion", "momentum", "volatility_clustering", "ma_distance_reversion"]
    )


def save_default_config(file_path: str = "user_config.json") -> None:
    """Save default configuration to file."""
    config = create_default_config()
    config.to_json_file(file_path)
    print(f"Default configuration saved to: {file_path}")


def load_config(file_path: str = "user_config.json") -> UserConfig:
    """Load configuration from file, creating default if not exists."""
    config_path = Path(file_path)
    
    if not config_path.exists():
        print(f"Configuration file {file_path} not found. Creating default configuration...")
        save_default_config(file_path)
        return create_default_config()
    
    try:
        config = UserConfig.from_json_file(file_path)
        issues = config.validate()
        
        if issues:
            print("⚠️  Configuration validation issues:")
            for issue in issues:
                print(f"  - {issue}")
            print("Using default configuration for problematic values...")
            
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration...")
        return create_default_config()


if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    config.print_summary()
    
    # Save default config
    save_default_config("example_user_config.json")
    
    # Load and validate config
    loaded_config = load_config("example_user_config.json")
    loaded_config.print_summary()
