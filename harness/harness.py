"""Harness — Safety layer that wraps every agent call.

The Harness is the 'manager' between the orchestrator and the agent.
It handles:
- Retry with exponential backoff + jitter
- Circuit breaker (3 failures → open → 30s cooldown → half-open)
- Timeout enforcement
- Response validation

This is what makes the workforce RELIABLE instead of fragile.
"""
import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable
from enum import Enum


class CircuitState(Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"           # Tripped — reject all calls
    HALF_OPEN = "HALF_OPEN"  # Probing — allow one test call


@dataclass
class HarnessStats:
    """Tracks harness safety metrics."""
    total_calls: int = 0
    successful: int = 0
    failed: int = 0
    retries: int = 0
    circuit_trips: int = 0
    timeouts: int = 0
    state: CircuitState = CircuitState.CLOSED
    
    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "successful": self.successful,
            "failed": self.failed,
            "retries": self.retries,
            "circuit_trips": self.circuit_trips,
            "timeouts": self.timeouts,
            "circuit_state": self.state.value,
        }


class CircuitBreaker:
    """Circuit breaker state machine.
    
    CLOSED → (3 failures) → OPEN → (30s) → HALF_OPEN → (success) → CLOSED
                                                   → (failure) → OPEN
    """
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
    
    def can_execute(self) -> bool:
        """Check if a call is allowed under current circuit state."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        return False
    
    def record_success(self):
        self._failure_count = 0
        self.state = CircuitState.CLOSED
    
    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class Harness:
    """Safety harness for agent execution.
    
    Wraps agent.execute() with:
    - Circuit breaker (prevents cascade failures)
    - Retry with exponential backoff + jitter
    - Timeout enforcement
    - Response validation
    """
    
    def __init__(self, max_retries: int = 3, timeout: float = 180.0,
                 failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.max_retries = max_retries
        self.timeout = timeout
        self.breaker = CircuitBreaker(failure_threshold, recovery_timeout)
        self.stats = HarnessStats()
    
    async def run(self, fn: Callable[[], Awaitable[str]], 
                  validate: Callable[[str], bool] = None) -> str:
        """Execute a function with full harness protection.
        
        Args:
            fn: Async function to call (e.g., lambda: agent.execute(task))
            validate: Optional validation function. Returns True if response is valid.
        
        Returns:
            Response string, or error message if all retries exhausted.
        """
        self.stats.total_calls += 1
        
        # Circuit breaker check
        if not self.breaker.can_execute():
            self.stats.failed += 1
            self.stats.circuit_trips += 1
            return "[HARNESS] Circuit breaker is OPEN. Service cooling down."
        
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await asyncio.wait_for(fn(), timeout=self.timeout)
                
                # Validate response if validator provided
                if validate and not validate(response):
                    raise ValueError("Response failed validation")
                
                self.breaker.record_success()
                self.stats.successful += 1
                return response
                
            except asyncio.TimeoutError:
                self.stats.timeouts += 1
                last_error = "TimeoutError"
                self.breaker.record_failure()
                if attempt < self.max_retries:
                    self.stats.retries += 1
                    delay = self._backoff_delay(attempt)
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                last_error = str(e) or type(e).__name__
                self.breaker.record_failure()
                if attempt < self.max_retries:
                    self.stats.retries += 1
                    delay = self._backoff_delay(attempt)
                    await asyncio.sleep(delay)
        
        self.stats.failed += 1
        return f"[HARNESS] All {self.max_retries} attempts failed. Last error: {last_error}"
    
    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter: base * 2^attempt + random(0, 1)."""
        base = 0.5
        return base * (2 ** (attempt - 1)) + random.uniform(0, 1)
    
    def get_stats(self) -> dict:
        return self.stats.summary()