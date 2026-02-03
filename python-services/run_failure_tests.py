"""
AURIX Failure Test Runner

Run failure simulations to validate system resilience.

Usage:
    python run_failure_tests.py
    python run_failure_tests.py --scenario "Redis Outage"
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from aurix.testing import FailureSimulator, FailureType, run_failure_tests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='AURIX Failure Simulation Tests')
    
    parser.add_argument('--scenario', type=str, 
                       help='Run specific scenario by name')
    parser.add_argument('--all', action='store_true', default=True,
                       help='Run all scenarios')
    parser.add_argument('--output', type=str, default='data/reports',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Run simulations
    simulator = FailureSimulator()
    
    if args.scenario:
        # Find matching scenario
        matching = [s for s in simulator.STANDARD_SCENARIOS 
                   if args.scenario.lower() in s.name.lower()]
        if not matching:
            logger.error(f"No scenario matching '{args.scenario}'")
            logger.info("Available scenarios:")
            for s in simulator.STANDARD_SCENARIOS:
                logger.info(f"  - {s.name}")
            sys.exit(1)
        
        for scenario in matching:
            simulator.run_scenario(scenario)
    else:
        simulator.run_all_scenarios()
    
    # Generate report
    report = simulator.generate_report()
    
    # Print report
    print("\n" + report)
    
    # Save report
    report_path = os.path.join(
        args.output, 
        f"failure_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    
    # Exit code based on results
    passed = sum(1 for r in simulator.results if r.passed)
    total = len(simulator.results)
    
    if passed == total:
        logger.info(f"✅ All {total} scenarios passed!")
        sys.exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} scenarios failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
