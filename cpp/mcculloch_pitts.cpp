/**
 * McCulloch-Pitts (M-P) Neuron From Scratch in Modern C++
 * University Hackathon Project 13: Sign Language Recognition and Translation Assistant
 * Author: Gehna
 * Date: 2026-10-28
 *
 * Compiles cleanly with: g++ -O3 cpp/mcculloch_pitts.cpp -o bin/mcculloch_pitts.exe
 */

#include <iostream>
#include <vector>
#include <cassert>
#include <string>

class McCullochPittsNeuron {
public:
    std::vector<int> weights;
    int threshold;
    bool has_inhibitory;

    McCullochPittsNeuron(const std::vector<int>& w, int th, bool inhibitory = false)
        : weights(w), threshold(th), has_inhibitory(inhibitory) {}

    int activate(const std::vector<int>& inputs) const {
        assert(inputs.size() == weights.size());

        // Inhibitory veto: if inhibitory input is active, output is strictly 0
        if (has_inhibitory && inputs.back() == 1) {
            return 0;
        }

        // Weighted sum of excitatory inputs
        int z = 0;
        size_t count = has_inhibitory ? inputs.size() - 1 : inputs.size();
        for (size_t i = 0; i < count; ++i) {
            z += weights[i] * inputs[i];
        }

        // Step function with threshold
        return (z >= threshold) ? 1 : 0;
    }
};

void run_test(const std::string& name, const McCullochPittsNeuron& neuron, const std::vector<int>& expected) {
    std::cout << "\nEvaluating " << name << " Gate:" << std::endl;
    std::cout << " x1 | x2 | Output" << std::endl;
    std::cout << "-----------------" << std::endl;

    std::vector<std::vector<int>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    for (size_t i = 0; i < inputs.size(); ++i) {
        int out = neuron.activate(inputs[i]);
        std::cout << "  " << inputs[i][0] << " |  " << inputs[i][1] << " |   " << out << std::endl;
        assert(out == expected[i]);
    }
    std::cout << "  -> Assertion PASSED: " << name << " truth table verified." << std::endl;
}

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << " C++ McCULLOCH-PITTS NEURON (Zero-Dependency STL)" << std::endl;
    std::cout << " University Requirement Compliance: Native C++ Module" << std::endl;
    std::cout << "=========================================================" << std::endl;

    // 1. AND Gate: weights=[1, 1], threshold=2
    McCullochPittsNeuron and_gate({1, 1}, 2);
    run_test("AND (theta=2)", and_gate, {0, 0, 0, 1});

    // 2. OR Gate: weights=[1, 1], threshold=1
    McCullochPittsNeuron or_gate({1, 1}, 1);
    run_test("OR (theta=1)", or_gate, {0, 1, 1, 1});

    // 3. AND-NOT Gate (x1 AND NOT x2): threshold=1, inhibitory=true
    McCullochPittsNeuron and_not_gate({1, 1}, 1, true);
    run_test("AND-NOT (x1 AND NOT x2)", and_not_gate, {0, 0, 1, 0});

    std::cout << "=========================================================" << std::endl;
    std::cout << " ALL C++ McCULLOCH-PITTS NEURON TESTS PASSED!" << std::endl;
    std::cout << "=========================================================" << std::endl;
    return 0;
}
