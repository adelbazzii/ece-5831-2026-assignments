from logic_gate import LogicGate


# XOR class that uses AND, NAND, and OR gates to compute the XOR output
class XorGate:
    def __init__(self, and_gate, nand_gate, or_gate):
        self.gate_type = "XOR"
        self.and_gate = and_gate
        self.nand_gate = nand_gate
        self.or_gate = or_gate

    def compute(self, inputs):
        # XOR output is computed as AND(NAND(inputs), OR(inputs))
        return self.and_gate.compute(
            [
                self.nand_gate.compute(inputs),
                self.or_gate.compute(inputs),
            ]
        )


# create objects of logic gates
and_gate = LogicGate("AND", [1, 1], -1.5)
nand_gate = LogicGate("NAND", [-1, -1], 1.5)
or_gate = LogicGate("OR", [1, 1], -0.5)
nor_gate = LogicGate("NOR", [-1, -1], 0.5)
xor_gate = XorGate(and_gate, nand_gate, or_gate)
