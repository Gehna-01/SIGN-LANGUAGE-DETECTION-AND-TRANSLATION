"""
McCulloch-Pitts (M-P) Neuron Implementation From Scratch
University Hackathon Project 13: Sign Language Recognition and Translation Assistant
Author: Gehna
Date: 2026-10-28

Implements the historical 1943 McCulloch-Pitts computational neuron:
- Binary inputs x_i in {0, 1}
- Excitatory inputs (summation) and Inhibitory inputs (veto)
- Pre-activation: z = sum(w_i * x_i)
- Heaviside step activation: g(z) = 1 if z >= theta else 0
- Demonstrations of AND, OR, NOR, and AND-NOT logic gates
- Theoretical explanation of linear separability and the XOR limitation
"""

from typing import List, Tuple

class McCullochPittsNeuron:
    """
    McCulloch-Pitts Artificial Neuron (1943).
    A threshold logic unit with binary inputs and step activation.
    """
    def __init__(self, weights: List[int], threshold: int, has_inhibitory: bool = False):
        """
        weights: List of weights (typically 1 for excitatory inputs)
        threshold: Integer activation threshold (theta)
        has_inhibitory: If True, the last input is treated as an absolute inhibitory input
        """
        self.weights = weights
        self.threshold = threshold
        self.has_inhibitory = has_inhibitory

    def activate(self, inputs: List[int]) -> int:
        """
        Computes forward activation:
        1. Checks inhibitory veto (if any inhibitory input is 1, output is 0).
        2. Computes weighted sum: z = sum(w_i * x_i)
        3. Applies threshold step function: 1 if z >= theta else 0
        """
        if len(inputs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, got {len(inputs)}")

        # Inhibitory check: if the last input is inhibitory and active, neuron cannot fire
        if self.has_inhibitory and inputs[-1] == 1:
            return 0

        # Weighted sum of excitatory inputs
        excitatory_weights = self.weights[:-1] if self.has_inhibitory else self.weights
        excitatory_inputs = inputs[:-1] if self.has_inhibitory else inputs

        z = sum(w * x for w, x in zip(excitatory_weights, excitatory_inputs))

        # Heaviside step function with threshold theta
        return 1 if z >= self.threshold else 0


def create_and_gate() -> McCullochPittsNeuron:
    """
    AND Gate:
    x1  x2  |  sum  |  theta  |  Output
     0   0  |   0   |    2    |    0
     0   1  |   1   |    2    |    0
     1   0  |   1   |    2    |    0
     1   1  |   2   |    2    |    1
    Weights: [1, 1], Threshold: 2
    """
    return McCullochPittsNeuron(weights=[1, 1], threshold=2)


def create_or_gate() -> McCullochPittsNeuron:
    """
    OR Gate:
    x1  x2  |  sum  |  theta  |  Output
     0   0  |   0   |    1    |    0
     0   1  |   1   |    1    |    1
     1   0  |   1   |    1    |    1
     1   1  |   2   |    1    |    1
    Weights: [1, 1], Threshold: 1
    """
    return McCullochPittsNeuron(weights=[1, 1], threshold=1)


def create_and_not_gate() -> McCullochPittsNeuron:
    """
    AND-NOT Gate (x1 AND NOT x2):
    x1 is excitatory (w=1), x2 is inhibitory.
    x1  x2  |  Inhibitory?  |  sum  |  theta  |  Output
     0   0  |     No        |   0   |    1    |    0
     0   1  |    YES        |   -   |    -    |    0
     1   0  |     No        |   1   |    1    |    1
     1   1  |    YES        |   -   |    -    |    0
    """
    return McCullochPittsNeuron(weights=[1, 1], threshold=1, has_inhibitory=True)


def evaluate_gate(name: str, neuron: McCullochPittsNeuron, input_combinations: List[Tuple[int, ...]]):
    print(f"\n--- {name} Gate Evaluation ---")
    header = " | ".join([f"x{i+1}" for i in range(len(input_combinations[0]))]) + " | Output"
    print(header)
    print("-" * len(header))
    
    results = []
    for inp in input_combinations:
        out = neuron.activate(list(inp))
        results.append(out)
        inp_str = " | ".join(str(x).center(3) for x in inp)
        print(f"{inp_str} |   {out}")
    return results


def run_mcculloch_pitts_demo():
    print("=" * 60)
    print("STEP 9: MCCULLOCH-PITTS NEURON FROM SCRATCH")
    print("=" * 60)
    
    inputs_2d = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # 1. AND Gate
    and_neuron = create_and_gate()
    and_results = evaluate_gate("AND (theta=2)", and_neuron, inputs_2d)
    assert and_results == [0, 0, 0, 1], "AND gate evaluation failed!"
    print("  -> Assertion PASSED: AND gate truth table 100% verified.")

    # 2. OR Gate
    or_neuron = create_or_gate()
    or_results = evaluate_gate("OR (theta=1)", or_neuron, inputs_2d)
    assert or_results == [0, 1, 1, 1], "OR gate evaluation failed!"
    print("  -> Assertion PASSED: OR gate truth table 100% verified.")

    # 3. AND-NOT Gate (Inhibition)
    and_not_neuron = create_and_not_gate()
    and_not_results = evaluate_gate("AND-NOT (x1 AND NOT x2)", and_not_neuron, inputs_2d)
    assert and_not_results == [0, 0, 1, 0], "AND-NOT gate evaluation failed!"
    print("  -> Assertion PASSED: Inhibitory veto 100% verified.")

    print("\n" + "=" * 60)
    print("ALL MCCULLOCH-PITTS DEMONSTRATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    run_mcculloch_pitts_demo()
