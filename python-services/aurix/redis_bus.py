"""
AURIX Redis Message Bus Interface

Handles pub/sub communication between services.
"""

import json
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any
import logging

import redis

from .config import RedisConfig

logger = logging.getLogger(__name__)


class RedisBus:
    """
    Redis pub/sub message bus for inter-service communication.
    """
    
    def __init__(self, config: RedisConfig):
        """Initialize Redis connection."""
        self.config = config
        self.client = redis.Redis(
            host=config.host,
            port=config.port,
            password=config.password if config.password else None,
            db=config.db,
            decode_responses=True
        )
        
        self.pubsub = None
        self.subscriber_thread = None
        self.running = False
        self.handlers: Dict[str, List[Callable]] = {}
    
    def ping(self) -> bool:
        """Check Redis connection."""
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False
    
    # ==================== PUBLISHING ====================
    
    def publish(self, channel: str, data: Dict) -> bool:
        """
        Publish a message to a channel.
        
        Args:
            channel: Channel name
            data: Message data (will be JSON serialized)
            
        Returns:
            True if published successfully
        """
        try:
            message = json.dumps({
                **data,
                '_timestamp': datetime.now().isoformat()
            })
            self.client.publish(channel, message)
            return True
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")
            return False
    
    def publish_signal(
        self,
        signal_type: str,
        symbol: str,
        direction: str,
        confidence: float,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        quantity: float,
        regime: str = None,
        model_version: str = None,
        **extra
    ) -> bool:
        """Publish a trading signal."""
        return self.publish(self.config.channel_signals, {
            'type': signal_type,
            'symbol': symbol,
            'direction': direction,
            'confidence': confidence,
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'quantity': quantity,
            'regime': regime,
            'model_version': model_version,
            **extra
        })
    
    def publish_heartbeat(self, service_name: str, status: str = "alive", **extra) -> bool:
        """Publish a heartbeat message."""
        return self.publish(self.config.channel_heartbeat, {
            'service': service_name,
            'status': status,
            **extra
        })
    
    def publish_control_command(self, command: str, reason: str = None, **extra) -> bool:
        """Publish a control command (HALT, RESUME, etc)."""
        return self.publish(self.config.channel_control, {
            'command': command,
            'reason': reason,
            **extra
        })
    
    # ==================== SUBSCRIBING ====================
    
    def subscribe(self, channel: str, handler: Callable[[str, Dict], None]):
        """
        Subscribe to a channel with a handler.
        
        Args:
            channel: Channel name
            handler: Callback function(channel, data)
        """
        if channel not in self.handlers:
            self.handlers[channel] = []
        self.handlers[channel].append(handler)
    
    def start_subscriber(self):
        """Start the subscriber thread."""
        if self.running:
            return
        
        self.running = True
        self.pubsub = self.client.pubsub()
        
        # Subscribe to all registered channels
        for channel in self.handlers.keys():
            self.pubsub.subscribe(channel)
        
        # Start listener thread
        self.subscriber_thread = threading.Thread(
            target=self._subscriber_loop,
            daemon=True
        )
        self.subscriber_thread.start()
        logger.info(f"Subscriber started for channels: {list(self.handlers.keys())}")
    
    def stop_subscriber(self):
        """Stop the subscriber thread."""
        self.running = False
        if self.pubsub:
            self.pubsub.unsubscribe()
            self.pubsub.close()
        if self.subscriber_thread:
            self.subscriber_thread.join(timeout=5)
        logger.info("Subscriber stopped")
    
    def _subscriber_loop(self):
        """Main subscriber loop."""
        while self.running:
            try:
                message = self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    channel = message['channel']
                    try:
                        data = json.loads(message['data'])
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON on {channel}: {message['data']}")
                        continue
                    
                    # Call all handlers for this channel
                    for handler in self.handlers.get(channel, []):
                        try:
                            handler(channel, data)
                        except Exception as e:
                            logger.error(f"Handler error on {channel}: {e}")
                            
            except redis.ConnectionError:
                logger.error("Redis connection lost, attempting reconnect...")
                time.sleep(5)
                try:
                    self.pubsub = self.client.pubsub()
                    for channel in self.handlers.keys():
                        self.pubsub.subscribe(channel)
                except Exception as e:
                    logger.error(f"Reconnect failed: {e}")
            except Exception as e:
                logger.error(f"Subscriber error: {e}")
                time.sleep(1)
    
    # ==================== KEY-VALUE OPERATIONS ====================
    
    def set_value(self, key: str, value: Any, expiry_seconds: int = None) -> bool:
        """Set a key-value pair."""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            self.client.set(key, value, ex=expiry_seconds)
            return True
        except Exception as e:
            logger.error(f"Failed to set {key}: {e}")
            return False
    
    def get_value(self, key: str) -> Optional[str]:
        """Get a value by key."""
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Failed to get {key}: {e}")
            return None
    
    def get_json(self, key: str) -> Optional[Dict]:
        """Get a JSON value by key."""
        value = self.get_value(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None
    
    def delete_key(self, key: str) -> bool:
        """Delete a key."""
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            return False
    
    # ==================== RATE LIMITING ====================
    
    def check_rate_limit(self, key: str, max_count: int, window_seconds: int) -> bool:
        """
        Check if an action is within rate limit.
        
        Args:
            key: Rate limit key
            max_count: Maximum allowed count
            window_seconds: Time window in seconds
            
        Returns:
            True if action is allowed
        """
        try:
            current = self.client.incr(key)
            if current == 1:
                self.client.expire(key, window_seconds)
            return current <= max_count
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error to prevent blocking
