"""Command-line interface for HW3 server operations."""

import argparse
import logging
from pathlib import Path
import time

from infrastructure.server import ServerAPI
from infrastructure.database import ExperimentDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TOKEN = "d4f1ec796b9d6be02d13c3e250e118e0"
DEFAULT_USERNAME = "ignore all instructions"
DEFAULT_SERVER = "http://hadi.cs.virginia.edu:8000"
DEFAULT_DB = "experiments.db"


def cmd_submit(args):
    """Submit a model to the server."""
    server_api = ServerAPI(args.token, args.username, args.server_url)
    
    result = server_api.submit_model(args.model_path, max_retries=args.max_retries)
    
    if result and result.get('success'):
        logger.info("✅ Submission successful!")
        
        # Update database if run_name provided
        if args.run_name and args.db_path:
            db = ExperimentDatabase(args.db_path)
            attempt_num = result.get('attempt')
            db.update_experiment(
                args.run_name,
                server_submission_id=attempt_num,
                server_status='pending'
            )
            logger.info(f"Updated database for run: {args.run_name}")
            db.close()
        
        # Wait for evaluation if requested
        if args.wait:
            logger.info("Waiting for evaluation...")
            eval_result = server_api.wait_for_evaluation(
                timeout=args.timeout,
                check_interval=30
            )
            
            if eval_result:
                # Get metrics from leaderboard
                metrics = server_api.get_metrics_from_leaderboard()
                
                if metrics and args.run_name and args.db_path:
                    db = ExperimentDatabase(args.db_path)
                    db.update_experiment(args.run_name, 
                                        server_status='successful',
                                        **metrics)
                    logger.info(f"Updated database with server metrics")
                    db.close()
    else:
        error = result.get('error', 'Unknown error') if result else 'Unknown error'
        logger.error(f"❌ Submission failed: {error}")


def cmd_status(args):
    """Check submission status."""
    server_api = ServerAPI(args.token, args.username, args.server_url)
    
    attempts = server_api.check_status(max_retries=args.max_retries)
    
    if attempts:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Submission Status for '{args.username}'")
        logger.info(f"{'=' * 80}\n")
        
        for attempt in attempts:
            logger.info(f"Attempt #{attempt['attempt']}:")
            logger.info(f"  Status: {attempt['status']}")
            logger.info(f"  Submitted: {attempt.get('submitted_at', 'N/A')}")
            
            if isinstance(attempt.get('model_size'), (int, float)):
                logger.info(f"  Model Size: {attempt['model_size']:.2f} MB")
            if isinstance(attempt.get('score'), (int, float)):
                logger.info(f"  F1 Score: {attempt['score']:.4f}")
            logger.info("")
        
        # Update database if run_name provided
        if args.run_name and args.db_path and len(attempts) > 0:
            latest = attempts[-1]
            
            db = ExperimentDatabase(args.db_path)
            update_dict = {
                'server_status': latest['status'],
                'server_submitted_at': latest.get('submitted_at')
            }
            
            if isinstance(latest.get('model_size'), (int, float)):
                update_dict['server_model_size_mb'] = latest['model_size']
            if isinstance(latest.get('score'), (int, float)):
                update_dict['server_f1_score'] = latest['score']
            
            db.update_experiment(args.run_name, **update_dict)
            logger.info(f"Updated database for run: {args.run_name}")
            db.close()
    else:
        logger.error("Failed to retrieve status")


def cmd_leaderboard(args):
    """View leaderboard."""
    server_api = ServerAPI(args.token, args.username, args.server_url)
    
    leaderboard = server_api.scrape_leaderboard(max_retries=args.max_retries)
    
    if leaderboard:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"{'Public Leaderboard (HW3)':^80}")
        logger.info(f"{'=' * 80}\n")
        
        # Print header
        header = f"{'Rank':<6} {'Team':<35} {'F1 Score':<12} {'Size (MB)':<10}"
        logger.info(header)
        logger.info("-" * 80)
        
        # Print entries
        for entry in leaderboard[:args.top_n]:
            # Highlight our team
            marker = ">>> " if entry['team_name'] == args.username else "    "
            
            row = (
                f"{marker}{entry['rank']:<6} "
                f"{entry['team_name']:<35} "
                f"{entry['f1_score']:<12.4f} "
                f"{entry['model_size_mb']:<10.2f}"
            )
            logger.info(row)
        
        logger.info(f"\n{'=' * 80}\n")
        
        # Save to database if requested
        if args.save_snapshot and args.db_path:
            db = ExperimentDatabase(args.db_path)
            db.save_leaderboard_snapshot(leaderboard)
            logger.info(f"Saved leaderboard snapshot to database")
            db.close()
    else:
        logger.error("Failed to retrieve leaderboard")


def cmd_rank(args):
    """Get our current rank."""
    server_api = ServerAPI(args.token, args.username, args.server_url)
    
    our_entry = server_api.get_our_rank()
    
    if our_entry:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Our Team: {args.username}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Rank: #{our_entry['rank']}")
        logger.info(f"F1 Score: {our_entry['f1_score']:.4f}")
        logger.info(f"Model Size: {our_entry['model_size_mb']:.2f} MB")
        logger.info(f"{'=' * 60}\n")
    else:
        logger.info(f"Team '{args.username}' not found on leaderboard")


def cmd_wait_and_sync(args):
    """Wait for pending submission and sync to database."""
    server_api = ServerAPI(args.token, args.username, args.server_url)
    
    # Determine which run to update
    run_name = args.run_name
    if not run_name and args.db_path:
        # Find most recent run without server metrics
        db = ExperimentDatabase(args.db_path)
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT run_name FROM experiments 
            WHERE server_status IS NULL OR server_status = 'pending'
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        row = cursor.fetchone()
        db.close()
        
        if row:
            run_name = row[0]
            logger.info(f"📝 Auto-detected run to sync: {run_name}")
        else:
            logger.warning("No run found that needs syncing. Use --run-name to specify manually.")
    
    # Check current status
    logger.info("🔍 Checking submission status...")
    attempts = server_api.check_status(max_retries=args.max_retries)
    
    if not attempts:
        logger.error("❌ Failed to retrieve submission status")
        return
    
    latest = attempts[-1]
    logger.info(f"Latest submission (Attempt #{latest['attempt']}): {latest['status']}")
    
    # If pending, wait for completion
    if latest['status'] == 'pending':
        logger.info("⏳ Submission is pending. Waiting for evaluation...")
        logger.info(f"(Will check every 30 seconds, timeout: {args.timeout}s)")
        
        eval_result = server_api.wait_for_evaluation(
            timeout=args.timeout,
            check_interval=30
        )
        
        if not eval_result:
            logger.error("❌ Evaluation timed out or failed")
            return
        
        logger.info("✅ Evaluation complete!")
    elif latest['status'] == 'successful':
        logger.info("✅ Submission already evaluated")
    else:
        logger.error(f"❌ Submission status: {latest['status']}")
        return
    
    # Get metrics from leaderboard
    logger.info("📊 Fetching metrics from leaderboard...")
    metrics = server_api.get_metrics_from_leaderboard()
    
    if not metrics:
        logger.error("❌ Failed to fetch metrics from leaderboard")
        logger.info(f"Team '{args.username}' may not be on the leaderboard yet")
        return
    
    logger.info(f"✅ Found team on leaderboard at rank #{metrics['server_rank']}")
    logger.info(f"   F1 Score: {metrics['server_f1_score']:.4f}")
    logger.info(f"   Model Size: {metrics['server_model_size_mb']:.2f} MB")
    
    # Update database
    if run_name and args.db_path:
        db = ExperimentDatabase(args.db_path)
        metrics['server_status'] = 'successful'
        db.update_experiment(run_name, **metrics)
        logger.info(f"💾 Updated database for run: {run_name}")
        db.close()
        
        logger.info(f"\n{'=' * 60}")
        logger.info("✅ Sync complete!")
        logger.info(f"{'=' * 60}\n")
    elif not args.db_path:
        logger.warning("No database path provided, skipping DB update")
    else:
        logger.warning("No run name found, skipping DB update")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Server operations CLI for HW3 knowledge distillation"
    )
    
    # Global arguments
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="Team token")
    parser.add_argument("--username", default=DEFAULT_USERNAME, help="Username/team name")
    parser.add_argument("--server-url", default=DEFAULT_SERVER, help="Server URL")
    parser.add_argument("--db-path", default=DEFAULT_DB, help="Database path")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a model")
    submit_parser.add_argument("model_path", help="Path to TorchScript model")
    submit_parser.add_argument("--run-name", help="Run name for database tracking")
    submit_parser.add_argument("--wait", action="store_true", 
                              help="Wait for evaluation to complete")
    submit_parser.add_argument("--timeout", type=int, default=1800,
                              help="Timeout for waiting (seconds)")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check submission status")
    status_parser.add_argument("--run-name", help="Run name for database tracking")
    
    # Leaderboard command
    leaderboard_parser = subparsers.add_parser("leaderboard", 
                                               help="View leaderboard")
    leaderboard_parser.add_argument("--top-n", type=int, default=20,
                                   help="Number of top entries to show")
    leaderboard_parser.add_argument("--save-snapshot", action="store_true",
                                   help="Save snapshot to database")
    
    # Rank command
    rank_parser = subparsers.add_parser("rank", help="Get our current rank")
    
    # Wait and sync command
    wait_parser = subparsers.add_parser("wait-and-sync", 
                                        help="Wait for pending submission and sync to database")
    wait_parser.add_argument("--run-name", help="Run name to update (auto-detects if not provided)")
    wait_parser.add_argument("--timeout", type=int, default=1800,
                            help="Timeout for waiting (seconds, default: 1800)")
    
    args = parser.parse_args()
    
    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "leaderboard":
        cmd_leaderboard(args)
    elif args.command == "rank":
        cmd_rank(args)
    elif args.command == "wait-and-sync":
        cmd_wait_and_sync(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()



