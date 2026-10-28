"""
Confidence-Gated Ambiguity Filtering with Temporal Consensus State Machine (CGAF-TCSM)
====================================================================================
Student Innovation Architecture for Real-Time Assistive Sign Language Translation.

Key Innovations:
1. Margin & Entropy Confidence Gating:
   - Suppresses ambiguous static predictions where top-1 vs top-2 probability
     margin delta (p_1 - p_2) < theta_margin, or normalized Shannon entropy H(p) > theta_entropy.
   - Prevents transient hand transition frames from corrupting translated text.

2. Temporal Consensus Window:
   - Maintains a sliding FIFO window of size K.
   - A character token is only emitted when >= C (consensus threshold) identical
     high-confidence predictions occur within the window.

3. Hysteresis Debounce & Operational Token State Machine:
   - Prevents duplicate character spamming (e.g., emits 'H-E-L-L-O' instead of 'HHHHHEEEEELLLLLLOOOOO').
   - Handles operational tokens ('space', 'del', 'nothing') with dedicated state transitions.
   - Requires a transition through 'nothing' or a distinct gesture before re-triggering the same character.
"""

import os
import sys
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict, Any


class ConfidenceGatedAmbiguityFilter:
    """
    Filters raw model output probabilities to suppress ambiguous predictions.
    
    Attributes:
        min_confidence (float): Minimum acceptable probability for top-1 class.
        margin_threshold (float): Minimum margin between top-1 and top-2 probabilities.
        entropy_threshold (float): Maximum allowable normalized Shannon entropy (0.0 to 1.0).
    """
    def __init__(
        self,
        min_confidence: float = 0.45,
        margin_threshold: float = 0.15,
        entropy_threshold: float = 0.70
    ):
        self.min_confidence = min_confidence
        self.margin_threshold = margin_threshold
        self.entropy_threshold = entropy_threshold

    def calculate_normalized_entropy(self, probabilities: np.ndarray) -> float:
        """
        Computes Shannon entropy normalized to [0, 1] range:
        H(p) = - sum(p * log(p)) / log(num_classes)
        """
        eps = 1e-12
        p = np.clip(probabilities, eps, 1.0)
        p = p / np.sum(p)
        raw_entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(len(probabilities))
        return float(raw_entropy / max_entropy)

    def filter_prediction(self, probabilities: np.ndarray) -> Tuple[Optional[int], float, Dict[str, Any]]:
        """
        Evaluates prediction probabilities against confidence, margin, and entropy gates.
        
        Returns:
            predicted_class (int or None): Class index if passed, else None (ambiguous/rejected).
            confidence (float): Top-1 probability.
            diagnostics (dict): Gate evaluation metrics.
        """
        num_classes = len(probabilities)
        sorted_indices = np.argsort(probabilities)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1]) if num_classes > 1 else top1_idx
        
        top1_prob = float(probabilities[top1_idx])
        top2_prob = float(probabilities[top2_idx]) if num_classes > 1 else 0.0
        margin = float(top1_prob - top2_prob)
        norm_entropy = self.calculate_normalized_entropy(probabilities)
        
        # Check gating conditions
        pass_confidence = top1_prob >= self.min_confidence
        pass_margin = margin >= self.margin_threshold
        pass_entropy = norm_entropy <= self.entropy_threshold
        
        is_accepted = pass_confidence and pass_margin and pass_entropy
        
        diagnostics = {
            "top1_class": top1_idx,
            "top1_prob": round(top1_prob, 4),
            "top2_class": top2_idx,
            "top2_prob": round(top2_prob, 4),
            "margin": round(margin, 4),
            "normalized_entropy": round(norm_entropy, 4),
            "pass_confidence": pass_confidence,
            "pass_margin": pass_margin,
            "pass_entropy": pass_entropy,
            "accepted": is_accepted
        }
        
        return (top1_idx if is_accepted else None), top1_prob, diagnostics


class TemporalConsensusStateMachine:
    """
    Temporal Consensus State Machine that aggregates frame-level predictions
    into stable transcribed text for assistive communication.
    
    Attributes:
        classes (list): List of 29 class label strings.
        window_size (int): Temporal buffer size (e.g., 5 frames).
        consensus_threshold (int): Minimum identical frames needed in buffer (e.g., 4 of 5).
        debounce_frames (int): Refractory period before repeating the same character.
    """
    def __init__(
        self,
        classes: List[str],
        window_size: int = 5,
        consensus_threshold: int = 4,
        debounce_frames: int = 3,
        gate_filter: Optional[ConfidenceGatedAmbiguityFilter] = None
    ):
        self.classes = classes
        self.window_size = window_size
        self.consensus_threshold = consensus_threshold
        self.debounce_frames = debounce_frames
        self.filter = gate_filter or ConfidenceGatedAmbiguityFilter()
        
        # Internal state buffers
        self.history_window = deque(maxlen=window_size)
        self.text_buffer = []
        self.last_committed_token = None
        self.frames_since_commit = 0
        self.current_state = "IDLE"  # IDLE, ACCUMULATING, COMMITTED

    def reset(self):
        """Resets the state machine buffers."""
        self.history_window.clear()
        self.text_buffer.clear()
        self.last_committed_token = None
        self.frames_since_commit = 0
        self.current_state = "IDLE"

    def process_frame(self, probabilities: np.ndarray) -> Dict[str, Any]:
        """
        Ingests a single video frame's softmax probability distribution and
        updates the state machine.
        
        Returns:
            step_result (dict): Emitted token, current text buffer, and diagnostic status.
        """
        filtered_class_idx, conf, diag = self.filter.filter_prediction(probabilities)
        
        # If ambiguous, append None to window
        token = self.classes[filtered_class_idx] if filtered_class_idx is not None else None
        self.history_window.append(token)
        self.frames_since_commit += 1
        
        emitted_token = None
        action_taken = "NONE"
        
        # Check for temporal consensus in sliding window
        valid_tokens = [t for t in self.history_window if t is not None]
        if valid_tokens:
            from collections import Counter
            counts = Counter(valid_tokens)
            dominant_token, dominant_count = counts.most_common(1)[0]
            
            # Consensus condition met
            if dominant_count >= self.consensus_threshold:
                # Check debounce condition (allow repetition only if passed debounce threshold or through nothing)
                can_commit = (
                    dominant_token != self.last_committed_token or
                    self.frames_since_commit >= self.debounce_frames or
                    dominant_token in ["del", "space"]
                )
                
                if can_commit:
                    emitted_token = dominant_token
                    self.last_committed_token = dominant_token
                    self.frames_since_commit = 0
                    
                    # Handle operational tokens
                    if dominant_token == "space":
                        if self.text_buffer and self.text_buffer[-1] != " ":
                            self.text_buffer.append(" ")
                            action_taken = "APPEND_SPACE"
                    elif dominant_token == "del":
                        if self.text_buffer:
                            removed = self.text_buffer.pop()
                            action_taken = f"DELETE_{removed}"
                    elif dominant_token == "nothing":
                        action_taken = "RESTING_STATE"
                    else:
                        # Standard alphabetic character A-Z
                        self.text_buffer.append(dominant_token)
                        action_taken = f"APPEND_CHAR_{dominant_token}"
                        
                    # Clear window after commit to prevent immediate re-triggering
                    self.history_window.clear()
                    
        return {
            "token_candidate": token,
            "emitted_token": emitted_token,
            "action_taken": action_taken,
            "accumulated_text": "".join(self.text_buffer),
            "window_state": list(self.history_window),
            "diagnostics": diag
        }

    def get_transcribed_text(self) -> str:
        """Returns the full accumulated transcribed text string."""
        return "".join(self.text_buffer)
