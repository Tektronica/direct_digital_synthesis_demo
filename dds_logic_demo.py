import numpy as np
import matplotlib.pyplot as plt


def run_demo():
    start()


def start():
    log = {"time": [], "phase": [], "address": [], "output": []}
    accumulator = FOUR_BIT_ACCUMULATOR()
    result, Cout = accumulator.four_bit_accumulator(1, '0100', Cin=0)
    print('first test', result)
    result, Cout = accumulator.four_bit_accumulator(1, '0100', Cin=Cout)
    print('second test', result)

    OBA = ONE_BIT_ACCUMULATOR(1)
    print(OBA.logic_one_bit_accumulator(clk=1, y=1, Cin=0))
    print(OBA.logic_one_bit_accumulator(clk=1, y=1, Cin=0))

    test = D_FLIP_FLOP_NAND(1)
    osc = 1
    for i in range(5):
        print(f'dff{i} and {osc}', test.logic_d_flip_flop(clk=osc, d=0))
        if osc == 0:
            osc = 1
        else:
            osc = 0

    osc = 1
    for i in range(3):
        print(f'dff{i} and {osc}', test.logic_d_flip_flop(clk=osc, d=1))
        if osc == 0:
            osc = 1
        else:
            osc = 0

    print('sr test')
    sr1 = SR_NAND_LATCH(1)
    print(sr1.logic_sr_nand_latch(0, 0))
    print(sr1.logic_sr_nand_latch(0, 1))
    print(sr1.logic_sr_nand_latch(1, 0))
    print(sr1.logic_sr_nand_latch(1, 1))


# LOGIC GATES ----------------------------------------------------------------------------------------------------------
def logic_not(A):
    return int(not A)


def logic_and(A, B):
    return A and B


def logic_or(A, B):
    return A or B


def logic_nor(A, B):
    A_plus_B = logic_or(A, B)
    result = logic_not(A_plus_B)
    return result


def logic_nand(A, B):
    AB = logic_and(A, B)
    result = logic_not(AB)
    return result


def logic_nand_xor(A, B):
    nAB = logic_nand(A, B)
    A_nand_nAB = logic_nand(A, nAB)
    B_nand_nAB = logic_nand(B, nAB)
    result = logic_nand(A_nand_nAB, B_nand_nAB)
    return result


# LOGIC BLOCKS ---------------------------------------------------------------------------------------------------------

# SR LATCH NOR LOGIC ---------------------------------------------------------------------------------------------------
class SR_NOR_LATCH:
    def __init__(self, logic_id=0):
        self.logic_id = logic_id
        self.Q = 0
        self.nQ = 1

    def get_id(self):
        return self.logic_id

    def logic_sr_nor_latch(self, R, S):
        Q = logic_nor(R, self.nQ)
        nQ = logic_nor(S, self.Q)
        return Q, nQ


class GATED_SR_NOR_LATCH:
    def __init__(self, logic_id=0):
        self.logic_id = logic_id
        self.sr1 = SR_NOR_LATCH(1)
        self.Q = 0
        self.nQ = 1

    def get_id(self):
        return self.logic_id

    def logic_gated_sr_latch(self, R, S, E):
        RE = logic_and(R, E)
        SE = logic_and(S, E)
        Q, nQ = self.sr1.logic_sr_nor_latch(RE, SE)

        return Q, nQ


# SR LATCH NAND LOGIC --------------------------------------------------------------------------------------------------
class SR_NAND_LATCH:
    def __init__(self, logic_id=0):
        self.logic_id = logic_id
        self.Q = 0
        self.nQ = 1

    def get_id(self):
        return self.logic_id

    def logic_sr_nand_latch(self, S, R):
        Q = logic_nand(S, self.nQ)
        nQ = logic_nand(R, self.Q)
        return Q, nQ


# class GATED_SR_NAND_LATCH:
#     def __init__(self, logic_id=0):
#         self.logic_id = logic_id
#         self.sr1 = SR_NAND_LATCH(1)
#         self.Q = 0
#         self.nQ = 0
#
#     def get_id(self):
#         return self.logic_id
#
#     def logic_gated_sr_latch(self, R, S, E):
#         SE = logic_nand(S, E)
#         RE = logic_nand(R, E)
#         Q, nQ = self.sr1.logic_sr_nand_latch(SE, RE)
#
#         return Q, nQ


# FLIP-FLOPS -----------------------------------------------------------------------------------------------------------
class D_FLIP_FLOP_NOR:
    def __init__(self, logic_id=0):
        self.logic_id = logic_id
        self.gsr1 = GATED_SR_NOR_LATCH(1)
        self.gsr2 = GATED_SR_NOR_LATCH(2)

    def logic_d_flip_flop(self, clk, d):
        # rising edge d-flip-flop
        nclk = logic_not(clk)
        nd = logic_not(d)

        Q, nQ = self.gsr1.logic_gated_sr_latch(R=d, S=nd, E=nclk)
        Qout, nQout = self.gsr2.logic_gated_sr_latch(R=Q, S=nQ, E=clk)

        return Qout, nQout


class D_FLIP_FLOP_NAND:
    """
    rising edge d-flip-flop implemented with NAND technology
    takes two rising edges to reach steady-state
    """

    def __init__(self, logic_id=0):
        self.logic_id = logic_id
        self.sr1 = SR_NAND_LATCH(1)
        self.sr2 = SR_NAND_LATCH(2)

        self.nQ0 = 1
        self.Q1 = 0
        self.nQ1 = 1

    def logic_d_flip_flop(self, clk=1, d=0):
        Q0, self.nQ0 = self.sr1.logic_sr_nand_latch(S=self.nQ1, R=clk)
        nQ0_and_clk = logic_and(self.nQ0, clk)
        Q, nQ = self.sr2.logic_sr_nand_latch(S=nQ0_and_clk, R=d)

        return Q, nQ


# ADDER ----------------------------------------------------------------------------------------------------------------
def logic_full_adder(A, B, Cin):
    xorAB = logic_nand_xor(A, B)

    # carry out
    A_and_B = logic_and(A, B)
    xorAB_and_Cin = logic_and(xorAB, Cin)
    Cout = logic_or(A_and_B, xorAB_and_Cin)

    # sum
    S = logic_nand_xor(xorAB, Cin)

    return S, Cout


# ACCUMULATOR BLOCK ----------------------------------------------------------------------------------------------------
class ONE_BIT_ACCUMULATOR:
    def __init__(self, logic_id=0):
        self.logic_id = logic_id
        self.dflipflop1 = D_FLIP_FLOP_NAND(1)
        self.Q = 0

    def logic_one_bit_accumulator(self, clk, y, Cin=0):
        # register of flip-flops
        s, Cout = logic_full_adder(A=y, B=self.Q, Cin=Cin)
        self.Q, Qn = self.dflipflop1.logic_d_flip_flop(clk, s)

        return s, Cout


class FOUR_BIT_ACCUMULATOR:
    def __init__(self, logic_id=0):
        self.logic_id = logic_id
        self.accumulator0 = ONE_BIT_ACCUMULATOR(1)
        self.accumulator1 = ONE_BIT_ACCUMULATOR(2)
        self.accumulator2 = ONE_BIT_ACCUMULATOR(3)
        self.accumulator3 = ONE_BIT_ACCUMULATOR(4)

    def four_bit_accumulator(self, clk=1, y='0000', Cin=0):
        y0 = int(y[0])
        y1 = int(y[1])
        y2 = int(y[2])
        y3 = int(y[3])

        # register of flip-flops
        s0, Cout0 = self.accumulator1.logic_one_bit_accumulator(clk=clk, y=y0, Cin=Cin)
        s1, Cout1 = self.accumulator1.logic_one_bit_accumulator(clk=clk, y=y1, Cin=Cout0)
        s2, Cout2 = self.accumulator1.logic_one_bit_accumulator(clk=clk, y=y2, Cin=Cout1)
        s3, Cout3 = self.accumulator1.logic_one_bit_accumulator(clk=clk, y=y3, Cin=Cout2)

        return str(s0) + str(s1) + str(s2) + str(s3), Cout3


# PLOT -----------------------------------------------------------------------------------------------------------------
def plot(log):
    # log
    time = log["time"]
    phase = log["phase"]
    address = log["address"]
    output = log["output"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle('Compressed Air Cannon', fontsize=18)

    ax1.plot(time, phase)
    ax1.set_title('phase')
    ax1.set_xlabel('barrel length (travelled) (cm)')
    ax1.set_ylabel('phase')
    ax1.grid()

    ax1.plot(time, output)
    ax2.set_title('output')
    ax2.set_xlabel('time (ns)')
    ax2.set_ylabel('amplitude')
    ax2.grid()


if __name__ == '__main__':
    run_demo()
