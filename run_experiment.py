"""Full orchestration: train → download → submit → sync.

Usage:
    # Full automated run
    python run_experiment.py --architecture shufflenet --epochs 50
    
    # With custom parameters
    python run_experiment.py --architecture mobilenet_v2 --temperature 6.0 --alpha 0.1 --epochs 100
    
    # Local training only (no Modal)
    python run_experiment.py --local --architecture shufflenet --epochs 20
"""

import argparse
import subprocess
import sys
import time
import logging
import re
from pathlib import Path
from datetime import datetime

from infrastructure.database import ExperimentDatabase
from infrastructure.server import ServerAPI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TOKEN = "d4f1ec796b9d6be02d13c3e250e118e0"
USERNAME = "ignore all instructions"
SERVER_URL = "http://hadi.cs.virginia.edu:8000"
DB_PATH = "experiments.db"


def run_modal_training(args) -> str:
    """Run training on Modal and return run name."""
    logger.info("="*80)
    logger.info("Step 1/4: Training on Modal A100")
    logger.info("="*80)
    
    # Build Modal command
    cmd = [
        "modal", "run", "modal_train.py",
        "--architecture", args.architecture,
        "--width-mult", str(args.width_mult),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--temperature", str(args.temperature),
        "--alpha", str(args.alpha),
        "--augmentation-strength", args.augmentation_strength,
    ]
    
    if args.use_weighted_hard_loss:
        cmd.append("--use-weighted-hard-loss")
    
    if args.run_name:
        cmd.extend(["--run-name", args.run_name])
    
    # Run Modal training and capture output
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Use Popen to stream output while also capturing it
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    output_lines = []
    run_name = None
    
    # Stream output and capture run name
    for line in process.stdout:
        print(line, end='')  # Print to terminal
        output_lines.append(line)
        
        # Look for run name pattern in Modal output
        # Pattern: "Run name: shufflenet_T4.0_A0.3_20251125_163422"
        if "Run name:" in line:
            match = re.search(r'Run name:\s*(\S+)', line)
            if match:
                run_name = match.group(1)
    
    process.wait()
    
    if process.returncode != 0:
        logger.error("❌ Modal training failed!")
        sys.exit(1)
    
    # Fallback: if run name not found in output, use provided or generate
    if not run_name:
        if args.run_name:
            run_name = args.run_name
        else:
            # Last resort: generate based on parameters
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{args.architecture}_T{args.temperature}_A{args.alpha}_{timestamp}"
            logger.warning(f"⚠️ Could not parse run name from Modal output, using generated: {run_name}")
    
    logger.info(f"✅ Training complete! Run name: {run_name}")
    return run_name


def download_model_from_modal(run_name: str, output_dir: str = "models") -> str:
    """Download model from Modal volume."""
    logger.info("="*80)
    logger.info("Step 2/4: Downloading model from Modal")
    logger.info("="*80)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Note: modal volume get uses volume-relative paths, not container paths
    modal_path = f"models/{run_name}/model.pt"
    local_path = f"{output_dir}/{run_name}.pt"
    
    # Download using modal volume get
    cmd = [
        "modal", "volume", "get",
        "project3-kd-data",
        modal_path,
        local_path
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"❌ Download failed: {result.stderr}")
        sys.exit(1)
    
    # Verify file exists
    if not Path(local_path).exists():
        logger.error(f"❌ Model file not found: {local_path}")
        sys.exit(1)
    
    size_mb = Path(local_path).stat().st_size / (1024 ** 2)
    logger.info(f"✅ Model downloaded: {local_path} ({size_mb:.2f} MB)")
    
    return local_path


def submit_to_server(model_path: str, run_name: str) -> bool:
    """Submit model to server."""
    logger.info("="*80)
    logger.info("Step 3/4: Submitting to server")
    logger.info("="*80)
    
    server_api = ServerAPI(TOKEN, USERNAME, SERVER_URL)
    
    # Submit
    result = server_api.submit_model(model_path, max_retries=3)
    
    if not result or not result.get('success'):
        logger.error(f"❌ Submission failed: {result.get('error') if result else 'Unknown error'}")
        return False
    
    logger.info("✅ Submission successful!")
    
    # Update database with submission info
    db = ExperimentDatabase(DB_PATH)
    db.update_experiment(
        run_name,
        server_submission_id=result.get('attempt'),
        server_status='pending',
        model_path=model_path
    )
    db.close()
    
    return True


def wait_and_sync(run_name: str, timeout: int = 1800) -> bool:
    """Wait for evaluation and sync results to database."""
    logger.info("="*80)
    logger.info("Step 4/4: Waiting for evaluation and syncing")
    logger.info("="*80)
    
    server_api = ServerAPI(TOKEN, USERNAME, SERVER_URL)
    
    # Wait for evaluation
    logger.info(f"⏳ Waiting for evaluation (timeout: {timeout}s)...")
    eval_result = server_api.wait_for_evaluation(
        timeout=timeout,
        check_interval=30
    )
    
    if not eval_result:
        logger.error("❌ Evaluation timed out or failed")
        return False
    
    logger.info("✅ Evaluation complete!")
    
    # Get metrics from leaderboard
    logger.info("📊 Fetching metrics from leaderboard...")
    time.sleep(5)  # Give server time to update leaderboard
    
    metrics = server_api.get_metrics_from_leaderboard()
    
    if not metrics:
        logger.error("❌ Failed to fetch metrics from leaderboard")
        return False
    
    logger.info(f"✅ Retrieved metrics:")
    logger.info(f"   Rank: #{metrics['server_rank']}")
    logger.info(f"   F1 Score: {metrics['server_f1_score']:.4f}")
    logger.info(f"   Model Size: {metrics['server_model_size_mb']:.2f} MB")
    
    # Update database
    db = ExperimentDatabase(DB_PATH)
    metrics['server_status'] = 'successful'
    db.update_experiment(run_name, **metrics)
    db.close()
    
    logger.info(f"💾 Database updated for run: {run_name}")
    
    return True


def create_experiment_in_db(run_name: str, args) -> None:
    """Create experiment record in database."""
    db = ExperimentDatabase(DB_PATH)
    
    config = {
        'student_architecture': args.architecture,
        'temperature': args.temperature,
        'alpha': args.alpha,
        'distillation_method': 'logit',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'augmentation_strategy': args.augmentation_strength,
        'use_class_weights': True,
        'use_weighted_hard_loss': args.use_weighted_hard_loss,
        'train_split': 0.9,
        'modal_run': not args.local,
        'gpu_type': 'H100' if not args.local else 'MPS',
        'notes': args.notes if hasattr(args, 'notes') else ''
    }
    
    try:
        db.create_experiment(run_name, config)
        logger.info(f"📝 Created experiment record: {run_name}")
    except Exception as e:
        logger.warning(f"Experiment may already exist: {e}")
    
    db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Full orchestration: train → download → submit → sync"
    )
    
    # Model configuration
    parser.add_argument("--architecture", default="shufflenet",
                       choices=["shufflenet", "shufflenet_custom", "mobilenet_v2", 
                               "mobilenet_v3_small", "mobilenet_v3_large"],
                       help="Student architecture")
    parser.add_argument("--width-mult", type=float, default=0.5,
                       help="Width multiplier for model")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size (128 default for H100)")
    parser.add_argument("--learning-rate", type=float, default=0.0003,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001,
                       help="Weight decay")
    
    # Distillation configuration
    parser.add_argument("--temperature", type=float, default=4.0,
                       help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.3,
                       help="Weight for hard loss")
    
    # Data configuration
    parser.add_argument("--augmentation-strength", default="light",
                       choices=["none", "light", "medium", "strong"],
                       help="Augmentation strength")
    parser.add_argument("--use-weighted-hard-loss", action="store_true",
                       help="Apply class weights to hard loss (for class imbalance)")
    
    # Execution configuration
    parser.add_argument("--run-name", type=str, default=None,
                       help="Custom run name")
    parser.add_argument("--local", action="store_true",
                       help="Train locally instead of Modal")
    parser.add_argument("--skip-submit", action="store_true",
                       help="Skip server submission")
    parser.add_argument("--timeout", type=int, default=1800,
                       help="Timeout for evaluation (seconds)")
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*80)
    logger.info("FULL EXPERIMENT ORCHESTRATION")
    logger.info("="*80)
    logger.info(f"Architecture: {args.architecture}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Alpha: {args.alpha}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Augmentation: {args.augmentation_strength}")
    logger.info(f"Weighted hard loss: {args.use_weighted_hard_loss}")
    logger.info(f"Training: {'Local' if args.local else 'Modal (H100)'}")
    logger.info("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        # Step 1: Train
        if args.local:
            logger.error("❌ Local training not yet implemented. Use Modal for now.")
            logger.info("   Run: modal run modal_train.py --architecture shufflenet")
            sys.exit(1)
        else:
            run_name = run_modal_training(args)
        
        # Create DB record
        create_experiment_in_db(run_name, args)
        
        # Step 2: Download
        model_path = download_model_from_modal(run_name)
        
        # Step 3: Submit (if not skipped)
        if not args.skip_submit:
            if not submit_to_server(model_path, run_name):
                logger.error("❌ Submission failed!")
                sys.exit(1)
            
            # Step 4: Wait and sync
            if not wait_and_sync(run_name, timeout=args.timeout):
                logger.error("❌ Sync failed!")
                sys.exit(1)
        else:
            logger.info("⏭️  Skipping server submission (--skip-submit)")
        
        # Final summary
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info("🎉 EXPERIMENT COMPLETE!")
        logger.info("="*80)
        logger.info(f"Run name: {run_name}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
        
        # Show leaderboard rank
        if not args.skip_submit:
            server_api = ServerAPI(TOKEN, USERNAME, SERVER_URL)
            our_entry = server_api.get_our_rank()
            if our_entry:
                logger.info(f"\n📊 Current Leaderboard Position:")
                logger.info(f"   Rank: #{our_entry['rank']}")
                logger.info(f"   F1 Score: {our_entry['f1_score']:.4f}")
                logger.info(f"   Model Size: {our_entry['model_size_mb']:.2f} MB")
        
        logger.info("="*80 + "\n")
        
    except KeyboardInterrupt:
        logger.info("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

