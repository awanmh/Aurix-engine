"""
AURIX Capital Validation Runner

Starts the 14-day testnet validation period.
Locks the model and monitors system behavior.

Usage:
    python run_validation.py --model-version v20240101_abc123
    python run_validation.py --resume state.json
"""

import argparse
import logging
import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Optional

from aurix.config import load_config
from aurix.db import Database
from aurix.redis_bus import RedisBus
from aurix.validation import (
    CapitalValidationMode,
    ValidationPhase,
    TrustLevel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/validation.log')
    ]
)
logger = logging.getLogger(__name__)


class ValidationRunner:
    """
    Runs the 14-day capital validation process.
    """
    
    def __init__(
        self,
        model_version: str,
        config_path: str = 'config/config.yaml',
        backtest_results_path: Optional[str] = None
    ):
        """
        Initialize validation runner.
        
        Args:
            model_version: Model version to validate
            config_path: Path to config file
            backtest_results_path: Path to backtest results JSON (for drift comparison)
        """
        self.config = load_config(config_path)
        self.db = Database(self.config.database)
        self.redis = RedisBus(self.config.redis)
        
        # Load backtest metrics if available
        backtest_metrics = None
        if backtest_results_path and os.path.exists(backtest_results_path):
            with open(backtest_results_path, 'r') as f:
                backtest_metrics = json.load(f)
            logger.info(f"Loaded backtest metrics from {backtest_results_path}")
        
        # Initialize validation mode
        self.validator = CapitalValidationMode(
            model_version=model_version,
            initial_equity=self.config.risk.initial_capital,
            backtest_metrics=backtest_metrics
        )
        
        self.running = False
        self.equity_check_interval = 60 * 15  # Check every 15 minutes
        self.daily_report_hour = 0  # Midnight UTC
        self.last_daily_report = None
    
    def start(self):
        """Start the validation process."""
        logger.info("=" * 60)
        logger.info("AURIX CAPITAL VALIDATION MODE STARTED")
        logger.info("=" * 60)
        logger.info(f"Model Version: {self.validator.model_version}")
        logger.info(f"Validation Period: {CapitalValidationMode.VALIDATION_DAYS} days")
        logger.info(f"Initial Equity: ${self.validator.initial_equity:,.2f}")
        logger.info("[WARNING] MODEL LEARNING IS DISABLED DURING VALIDATION")
        logger.info("=" * 60)
        
        self.running = True
        
        # Subscribe to trade updates
        self.redis.subscribe(
            self.config.redis.channel_signals,
            self._handle_trade_signal
        )
        self.redis.start_subscriber()
        
        # Main monitoring loop
        self._run_monitoring_loop()
    
    def stop(self):
        """Stop validation process."""
        logger.info("Stopping validation...")
        self.running = False
        self.redis.stop_subscriber()
        
        # Save final state
        self.validator.save_state('data/validation/final_state.json')
        
        # Generate final report
        final_report = self.validator.generate_daily_report()
        logger.info("\n" + final_report)
        
        with open('data/validation/final_report.txt', 'w') as f:
            f.write(final_report)
    
    def _run_monitoring_loop(self):
        """Main monitoring loop."""
        os.makedirs('data/validation', exist_ok=True)
        
        last_equity_check = datetime.now()
        
        while self.running:
            try:
                now = datetime.now()
                
                # Periodic equity check
                if (now - last_equity_check).seconds >= self.equity_check_interval:
                    self._check_equity()
                    last_equity_check = now
                
                # Daily report
                if now.hour == self.daily_report_hour:
                    if self.last_daily_report is None or self.last_daily_report.date() < now.date():
                        self._generate_daily_report()
                        self.last_daily_report = now
                
                # Check if we should halt
                should_halt, reason = self.validator.should_halt()
                if should_halt:
                    logger.error(f"VALIDATION HALT: {reason}")
                    self._trigger_halt(reason)
                    break
                
                # Check if validation complete
                state = self.validator.get_state()
                if state.phase == ValidationPhase.COMPLETE:
                    logger.info("Validation period complete!")
                    self._complete_validation()
                    break
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(60)
    
    def _check_equity(self):
        """Check current equity and record it."""
        try:
            # Get current account state from database
            account = self.db.get_account_state()
            if account:
                equity = account.get('equity', self.validator.initial_equity)
                self.validator.record_equity(equity)
                
                state = self.validator.get_state()
                logger.info(
                    f"[Day {state.days_elapsed}/{CapitalValidationMode.VALIDATION_DAYS}] "
                    f"Equity: ${equity:,.2f} | CTS: {state.trust_score.total_score:.1f} | "
                    f"DD: {state.equity_metrics.current_drawdown_pct:.1%}"
                )
        except Exception as e:
            logger.error(f"Failed to check equity: {e}")
    
    def _handle_trade_signal(self, channel: str, data: dict):
        """Handle trade completion signals."""
        if data.get('type') != 'TRADE_COMPLETE':
            return
        
        try:
            pnl = data.get('net_pnl', 0)
            confidence = data.get('confidence', 0.5)
            was_win = pnl > 0
            
            self.validator.record_trade(pnl, confidence, was_win)
            
            logger.info(f"Trade recorded: PnL=${pnl:.2f}, Confidence={confidence:.1%}")
            
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
    
    def _generate_daily_report(self):
        """Generate and save daily report."""
        report = self.validator.generate_daily_report()
        
        # Log it
        logger.info("\n" + report)
        
        # Save to file
        date_str = datetime.now().strftime('%Y%m%d')
        report_path = f'data/validation/daily_report_{date_str}.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save state
        state_path = f'data/validation/state_{date_str}.json'
        self.validator.save_state(state_path)
        
        logger.info(f"Daily report saved to {report_path}")
    
    def _trigger_halt(self, reason: str):
        """Trigger trading halt due to validation failure."""
        logger.critical(f"TRIGGERING VALIDATION HALT: {reason}")
        
        # Publish halt command
        self.redis.publish_control_command(
            command="HALT",
            reason=f"Validation halt: {reason}"
        )
        
        # Save state
        self.validator.save_state('data/validation/halt_state.json')
        
        # Generate halt report
        report = self.validator.generate_daily_report()
        with open('data/validation/halt_report.txt', 'w') as f:
            f.write(f"HALT REASON: {reason}\n\n")
            f.write(report)
    
    def _complete_validation(self):
        """Handle validation completion."""
        state = self.validator.get_state()
        
        logger.info("=" * 60)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 60)
        
        if state.trust_score.trust_level == TrustLevel.VALIDATED:
            logger.info("✅ SYSTEM VALIDATED - Ready for live deployment")
            logger.info("   Recommended: Start with 25% of intended capital")
        elif state.trust_score.trust_level == TrustLevel.HIGH:
            logger.info("⚠️ HIGH TRUST - Consider extended validation")
            logger.info("   Recommended: Run additional 7 days before live")
        elif state.trust_score.trust_level == TrustLevel.MEDIUM:
            logger.info("⚠️ MEDIUM TRUST - Review issues before proceeding")
            logger.info("   Recommended: Do not deploy to live trading")
        else:
            logger.info("❌ LOW TRUST - Do not deploy")
            logger.info("   Recommended: Review and fix system issues")
        
        logger.info(f"\nFinal CTS: {state.trust_score.total_score:.1f}/100")
        logger.info(f"Total Trades: {self.validator.trade_count}")
        logger.info(f"Final Return: {state.equity_metrics.total_return_pct:.1%}")
        logger.info(f"Max Drawdown: {state.equity_metrics.max_drawdown_pct:.1%}")
        
        # Save final report
        self.stop()


def main():
    parser = argparse.ArgumentParser(description='AURIX Capital Validation Mode')
    
    parser.add_argument('--model-version', type=str, required=True,
                       help='Model version to validate')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Config file path')
    parser.add_argument('--backtest-results', type=str,
                       help='Path to backtest results JSON for drift comparison')
    parser.add_argument('--resume', type=str,
                       help='Resume from saved state file')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/validation', exist_ok=True)
    
    # Create runner
    runner = ValidationRunner(
        model_version=args.model_version,
        config_path=args.config,
        backtest_results_path=args.backtest_results
    )
    
    # Handle signals
    import signal
    def signal_handler(sig, frame):
        runner.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start validation
    try:
        runner.start()
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    # Fix import for Optional
    from typing import Optional
    main()
