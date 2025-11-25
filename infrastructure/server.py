"""Server API for HW3 submission and status checking."""

import time
import requests
from typing import Optional, Dict, Any, List
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class ServerAPI:
    """API client for HW3 submission server."""
    
    def __init__(self, token: str, username: str, server_url: str = "http://hadi.cs.virginia.edu:8000"):
        """Initialize server API client.
        
        Args:
            token: User authentication token
            username: Username for leaderboard
            server_url: Base server URL (default: http://hadi.cs.virginia.edu:8000)
        """
        self.token = token
        self.username = username
        self.server_url = server_url.rstrip('/')
    
    def submit_model(self, model_path: str, max_retries: int = 3, 
                     wait_on_rate_limit: bool = True, 
                     rate_limit_wait_minutes: int = 16,
                     max_rate_limit_retries: int = 5) -> Optional[Dict[str, Any]]:
        """Submit a model to the server with automatic rate limit handling.
        
        Args:
            model_path: Path to TorchScript model (.pt file)
            max_retries: Maximum number of retry attempts for network errors
            wait_on_rate_limit: If True, wait and retry when rate limited
            rate_limit_wait_minutes: Minutes to wait when rate limited (default: 16)
            max_rate_limit_retries: Max times to retry after rate limit (default: 5)
            
        Returns:
            Dictionary with submission result or None on failure
        """
        url = f"{self.server_url}/submit"
        rate_limit_count = 0
        
        while rate_limit_count <= max_rate_limit_retries:
            for attempt in range(max_retries):
                try:
                    with open(model_path, 'rb') as f:
                        files = {'file': f}
                        data = {'token': self.token}
                        
                        logger.info(f"Submitting model to {url} (attempt {attempt + 1}/{max_retries})...")
                        response = requests.post(url, data=data, files=files, timeout=60)
                        
                        if response.status_code == 200:
                            result = response.json()
                            logger.info(f"✅ Submission successful!")
                            logger.info(f"   Message: {result.get('message', 'No message')}")
                            return {
                                'success': True,
                                'message': result.get('message'),
                                'attempt': result.get('attempt'),
                                **result
                            }
                        elif response.status_code == 429:
                            # Rate limit hit
                            rate_limit_count += 1
                            if wait_on_rate_limit and rate_limit_count <= max_rate_limit_retries:
                                wait_seconds = rate_limit_wait_minutes * 60
                                logger.warning(f"⚠️  Rate limit hit (attempt {rate_limit_count}/{max_rate_limit_retries}).")
                                logger.info(f"⏳ Waiting {rate_limit_wait_minutes} minutes before retrying...")
                                
                                # Show countdown every minute
                                for remaining in range(rate_limit_wait_minutes, 0, -1):
                                    logger.info(f"   {remaining} minutes remaining...")
                                    time.sleep(60)
                                
                                logger.info(f"🔄 Retrying submission after rate limit wait...")
                                break  # Break inner loop to retry submission
                            else:
                                logger.error(f"❌ Rate limit exceeded max retries ({max_rate_limit_retries})")
                                return {
                                    'success': False,
                                    'error': 'Rate limit exceeded. Max retries reached.',
                                    'status_code': 429
                                }
                        else:
                            logger.error(f"❌ Submission failed with status {response.status_code}")
                            logger.error(f"   Response: {response.text}")
                            
                            if attempt < max_retries - 1:
                                wait_time = 2 ** attempt  # Exponential backoff
                                logger.info(f"Retrying in {wait_time} seconds...")
                                time.sleep(wait_time)
                            else:
                                return {
                                    'success': False,
                                    'error': response.text,
                                    'status_code': response.status_code
                                }
                
                except requests.exceptions.Timeout:
                    logger.error(f"❌ Request timed out")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying...")
                        time.sleep(2 ** attempt)
                except Exception as e:
                    logger.error(f"❌ Error submitting model: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        return {
                            'success': False,
                            'error': str(e)
                        }
            else:
                # Inner loop completed without break (no rate limit), exit outer loop
                break
        
        return None
    
    def check_status(self, max_retries: int = 3) -> Optional[List[Dict[str, Any]]]:
        """Check submission status for this token.
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of submission attempts or None on failure
        """
        url = f"{self.server_url}/submission-status/{self.token}"
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Checking status at {url}...")
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    attempts = response.json()
                    return attempts
                elif response.status_code == 404:
                    logger.warning(f"⚠️  No submissions found for token {self.token}")
                    return []
                else:
                    logger.error(f"❌ Status check failed with status {response.status_code}")
                    logger.error(f"   Response: {response.text}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
            
            except requests.exceptions.Timeout:
                logger.error(f"❌ Request timed out")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"❌ Error checking status: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def scrape_leaderboard(self, max_retries: int = 3) -> Optional[List[Dict[str, Any]]]:
        """Scrape the public leaderboard.
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of leaderboard entries or None on failure
        """
        url = f"{self.server_url}/leaderboard3"
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching leaderboard from {url}...")
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find the leaderboard table
                    table = soup.find('table')
                    if not table:
                        logger.warning("⚠️  No leaderboard table found in HTML")
                        return []
                    
                    entries = []
                    rows = table.find_all('tr')[1:]  # Skip header row
                    
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 4:  # rank, team, f1, size
                            try:
                                entry = {
                                    'rank': int(cols[0].text.strip()),
                                    'team_name': cols[1].text.strip(),
                                    'f1_score': float(cols[2].text.strip()),
                                    'model_size_mb': float(cols[3].text.strip())
                                }
                                entries.append(entry)
                            except (ValueError, IndexError) as e:
                                logger.warning(f"⚠️  Error parsing row: {e}")
                                continue
                    
                    logger.info(f"✅ Fetched {len(entries)} leaderboard entries")
                    return entries
                else:
                    logger.error(f"❌ Leaderboard fetch failed with status {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
            
            except requests.exceptions.Timeout:
                logger.error(f"❌ Request timed out")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"❌ Error scraping leaderboard: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def get_our_rank(self) -> Optional[Dict[str, Any]]:
        """Get our team's entry from the leaderboard.
        
        Returns:
            Dictionary with our team's data or None if not found
        """
        leaderboard = self.scrape_leaderboard()
        if not leaderboard:
            return None
        
        for entry in leaderboard:
            if entry['team_name'] == self.username:
                return entry
        
        return None
    
    def wait_for_evaluation(self, timeout: int = 1800, check_interval: int = 30) -> bool:
        """Wait for a pending submission to be evaluated.
        
        Args:
            timeout: Maximum time to wait in seconds (default: 30 minutes)
            check_interval: Time between status checks in seconds
            
        Returns:
            True if evaluation completed successfully, False otherwise
        """
        start_time = time.time()
        
        logger.info(f"⏳ Waiting for evaluation (timeout: {timeout}s, check every {check_interval}s)")
        
        while (time.time() - start_time) < timeout:
            attempts = self.check_status()
            
            if not attempts:
                logger.warning("⚠️  Could not check status, retrying...")
                time.sleep(check_interval)
                continue
            
            latest = attempts[-1]
            status = latest.get('status', 'unknown')
            
            if status == 'successful':
                logger.info(f"✅ Evaluation complete!")
                return True
            elif status == 'failed':
                logger.error(f"❌ Evaluation failed")
                return False
            elif status == 'pending':
                elapsed = int(time.time() - start_time)
                logger.info(f"   Still pending... ({elapsed}s elapsed)")
                time.sleep(check_interval)
            else:
                logger.warning(f"⚠️  Unknown status: {status}")
                time.sleep(check_interval)
        
        logger.error(f"❌ Timeout waiting for evaluation")
        return False
    
    def get_metrics_from_leaderboard(self) -> Optional[Dict[str, Any]]:
        """Get our metrics from the leaderboard.
        
        Returns:
            Dictionary with server metrics or None if not found
        """
        our_entry = self.get_our_rank()
        
        if not our_entry:
            return None
        
        return {
            'server_rank': our_entry['rank'],
            'server_f1_score': our_entry['f1_score'],
            'server_model_size_mb': our_entry['model_size_mb']
        }

