"""Database management for experiment tracking."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager


class ExperimentDatabase:
    """SQLite database for tracking knowledge distillation experiments."""
    
    def __init__(self, db_path: str = "experiments.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Main experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                
                -- Model architecture
                student_architecture TEXT,
                teacher_model TEXT DEFAULT 'google/medsiglip-448',
                
                -- Distillation hyperparameters
                temperature REAL,
                alpha REAL,
                distillation_method TEXT,  -- 'logit', 'feature', 'hybrid'
                
                -- Training hyperparameters
                epochs INTEGER,
                batch_size INTEGER,
                learning_rate REAL,
                weight_decay REAL,
                optimizer TEXT,
                scheduler TEXT,
                
                -- Data augmentation
                augmentation_strategy TEXT,
                use_class_weights BOOLEAN,
                train_split REAL,
                
                -- Training metrics
                best_epoch INTEGER,
                best_val_loss REAL,
                best_val_f1 REAL,
                best_val_accuracy REAL,
                final_train_loss REAL,
                final_val_loss REAL,
                
                -- Model info
                num_parameters INTEGER,
                model_size_mb REAL,
                
                -- Modal/compute info
                modal_run BOOLEAN DEFAULT TRUE,
                gpu_type TEXT,
                training_duration_seconds REAL,
                
                -- Server submission info
                server_submission_id INTEGER,
                server_status TEXT,  -- 'pending', 'successful', 'failed'
                server_submitted_at TEXT,
                server_model_size_mb REAL,
                server_f1_score REAL,
                server_accuracy REAL,
                server_rank INTEGER,
                
                -- File paths
                model_path TEXT,
                checkpoint_path TEXT,
                config_path TEXT,
                
                -- Notes
                notes TEXT
            )
        """)
        
        # Per-epoch training history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                train_loss REAL,
                val_loss REAL,
                val_f1 REAL,
                val_accuracy REAL,
                learning_rate REAL,
                timestamp TEXT,
                FOREIGN KEY (run_id) REFERENCES experiments(run_id),
                UNIQUE(run_id, epoch)
            )
        """)
        
        # Per-class metrics table (for detailed analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS per_class_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                class_idx INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                f1_score REAL,
                precision_score REAL,
                recall REAL,
                support INTEGER,
                FOREIGN KEY (run_id) REFERENCES experiments(run_id),
                UNIQUE(run_id, class_idx)
            )
        """)
        
        # Leaderboard snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                rank INTEGER,
                team_name TEXT,
                f1_score REAL,
                accuracy REAL,
                model_size_mb REAL
            )
        """)
        
        self.conn.commit()
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
    
    def create_experiment(self, run_name: str, config: Dict[str, Any]) -> int:
        """Create a new experiment record.
        
        Args:
            run_name: Unique name for this run
            config: Configuration dictionary
            
        Returns:
            run_id of created experiment
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO experiments (
                    run_name, timestamp,
                    student_architecture, teacher_model,
                    temperature, alpha, distillation_method,
                    epochs, batch_size, learning_rate, weight_decay,
                    optimizer, scheduler,
                    augmentation_strategy, use_class_weights, train_split,
                    modal_run, gpu_type,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_name,
                datetime.now().isoformat(),
                config.get('student_architecture'),
                config.get('teacher_model', 'google/medsiglip-448'),
                config.get('temperature'),
                config.get('alpha'),
                config.get('distillation_method', 'logit'),
                config.get('epochs'),
                config.get('batch_size'),
                config.get('learning_rate'),
                config.get('weight_decay'),
                config.get('optimizer', 'Adam'),
                config.get('scheduler'),
                config.get('augmentation_strategy'),
                config.get('use_class_weights', False),
                config.get('train_split', 0.9),
                config.get('modal_run', True),
                config.get('gpu_type', 'A100'),
                config.get('notes')
            ))
            
            run_id = cursor.lastrowid
            return run_id
    
    def update_experiment(self, run_name: str, **kwargs):
        """Update experiment fields.
        
        Args:
            run_name: Name of experiment to update
            **kwargs: Fields to update
        """
        if not kwargs:
            return
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            # Build update query
            set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
            values = list(kwargs.values()) + [run_name]
            
            cursor.execute(f"""
                UPDATE experiments
                SET {set_clause}
                WHERE run_name = ?
            """, values)
    
    def add_epoch_history(self, run_name: str, epoch: int, metrics: Dict[str, float]):
        """Add per-epoch training history.
        
        Args:
            run_name: Name of experiment
            epoch: Epoch number
            metrics: Dictionary of metrics for this epoch
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            # Get run_id
            cursor.execute("SELECT run_id FROM experiments WHERE run_name = ?", (run_name,))
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Experiment {run_name} not found")
            run_id = result[0]
            
            cursor.execute("""
                INSERT OR REPLACE INTO training_history (
                    run_id, epoch,
                    train_loss, val_loss, val_f1, val_accuracy,
                    learning_rate, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, epoch,
                metrics.get('train_loss'),
                metrics.get('val_loss'),
                metrics.get('val_f1'),
                metrics.get('val_accuracy'),
                metrics.get('learning_rate'),
                datetime.now().isoformat()
            ))
    
    def add_per_class_metrics(self, run_name: str, class_metrics: List[Dict[str, Any]]):
        """Add per-class metrics.
        
        Args:
            run_name: Name of experiment
            class_metrics: List of per-class metric dictionaries
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            # Get run_id
            cursor.execute("SELECT run_id FROM experiments WHERE run_name = ?", (run_name,))
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Experiment {run_name} not found")
            run_id = result[0]
            
            for cls_metric in class_metrics:
                cursor.execute("""
                    INSERT OR REPLACE INTO per_class_metrics (
                        run_id, class_idx, class_name,
                        f1_score, precision_score, recall, support
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    cls_metric['class_idx'],
                    cls_metric['class_name'],
                    cls_metric.get('f1_score'),
                    cls_metric.get('precision'),
                    cls_metric.get('recall'),
                    cls_metric.get('support')
                ))
    
    def get_experiment(self, run_name: str) -> Optional[Dict[str, Any]]:
        """Get experiment by name.
        
        Args:
            run_name: Name of experiment
            
        Returns:
            Dictionary of experiment data or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE run_name = ?", (run_name,))
        result = cursor.fetchone()
        return dict(result) if result else None
    
    def get_recent_experiments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent experiments.
        
        Args:
            limit: Number of experiments to return
            
        Returns:
            List of experiment dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM experiments
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_best_experiments(self, metric: str = 'server_f1_score', limit: int = 10) -> List[Dict[str, Any]]:
        """Get best experiments by a metric.
        
        Args:
            metric: Metric to sort by
            limit: Number of experiments to return
            
        Returns:
            List of experiment dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT * FROM experiments
            WHERE {metric} IS NOT NULL
            ORDER BY {metric} DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_training_history(self, run_name: str) -> List[Dict[str, Any]]:
        """Get training history for an experiment.
        
        Args:
            run_name: Name of experiment
            
        Returns:
            List of epoch history dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT h.* FROM training_history h
            JOIN experiments e ON h.run_id = e.run_id
            WHERE e.run_name = ?
            ORDER BY h.epoch
        """, (run_name,))
        return [dict(row) for row in cursor.fetchall()]
    
    def save_leaderboard_snapshot(self, leaderboard_data: List[Dict[str, Any]]):
        """Save a snapshot of the leaderboard.
        
        Args:
            leaderboard_data: List of leaderboard entries
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            for entry in leaderboard_data:
                cursor.execute("""
                    INSERT INTO leaderboard_snapshots (
                        timestamp, rank, team_name,
                        f1_score, accuracy, model_size_mb
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    entry.get('rank'),
                    entry.get('team_name'),
                    entry.get('f1_score'),
                    entry.get('accuracy'),
                    entry.get('model_size_mb')
                ))
    
    def export_to_csv(self, output_path: str):
        """Export experiments table to CSV.
        
        Args:
            output_path: Path to output CSV file
        """
        import pandas as pd
        
        df = pd.read_sql_query("SELECT * FROM experiments", self.conn)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} experiments to {output_path}")
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()




