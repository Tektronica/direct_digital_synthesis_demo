import numpy as np
import matplotlib.pyplot as plt


def run_demo():
    start()


def start():
    log = {"time": [], "phase": [], "address": [], "output": []}

    print('D Flip Flop test')
    dff = D_FLIP_FLOP()
    print(1, '---->', dff.logic_d_flip_flop(clk=0, d=1), 'first access (no change)')
    print(2, 'rising edge ---->', dff.logic_d_flip_flop(clk=1, d=1), 'should change to 1')
    print(3, '---->', dff.logic_d_flip_flop(clk=0, d=0), 'no change')
    print(4, 'rising edge ---->', dff.logic_d_flip_flop(clk=1, d=0), 'change to 0')
    print(5, '---->', dff.logic_d_flip_flop(clk=0, d=1), 'no change')
    print(6, 'rising edge ---->', dff.logic_d_flip_flop(clk=1, d=1), 'change to 1')

    print('\nfull adder test')
    print(logic_full_adder(A=0, B=0, Cin=0), '--> sum = 0, Cout = 0')
    print(logic_full_adder(A=0, B=0, Cin=1), '--> sum = 1, Cout = 0')
    print(logic_full_adder(A=0, B=1, Cin=0), '--> sum = 1, Cout = 0')
    print(logic_full_adder(A=0, B=1, Cin=1), '--> sum = 0, Cout = 1')
    print(logic_full_adder(A=1, B=0, Cin=0), '--> sum = 1, Cout = 0')
    print(logic_full_adder(A=1, B=0, Cin=1), '--> sum = 0, Cout = 1')
    print(logic_full_adder(A=1, B=1, Cin=0), '--> sum = 0, Cout = 1')
    print(logic_full_adder(A=1, B=1, Cin=1), '--> sum = 1, Cout = 1')

    print('\nfour bit accumulator')
    FBA = FOUR_BIT_ACCUMULATOR()
    print(FBA.four_bit_accumulator(clk=1, y='0010', Cin=0), '--> sum = 0010')
    print('\tfalling-edge', FBA.four_bit_accumulator(clk=0, y='0000', Cin=0), '--> same: sum = 0010')
    print(FBA.four_bit_accumulator(clk=1, y='0100', Cin=0), '--> sum = 0110')
    print('\tfalling-edge', FBA.four_bit_accumulator(clk=0, y='0000', Cin=0), '--> same: sum = 0110')
    print(FBA.four_bit_accumulator(clk=1, y='1000', Cin=0), '--> sum = 1110')
    print('\tfalling-edge', FBA.four_bit_accumulator(clk=0, y='0000', Cin=0), '--> same: sum = 1110')
    print(FBA.four_bit_accumulator(clk=1, y='0001', Cin=0), '--> sum = 1111')
    print('\tfalling-edge', FBA.four_bit_accumulator(clk=0, y='0000', Cin=0), '--> same: sum = 1110')
    print(FBA.four_bit_accumulator(clk=1, y='0001', Cin=0), '-->sum = 1110 with carry')


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


def xor(A, B):
    nAB = logic_nand(A, B)
    A_nand_nAB = logic_nand(A, nAB)
    B_nand_nAB = logic_nand(B, nAB)
    result = logic_nand(A_nand_nAB, B_nand_nAB)
    return result


# LOGIC BLOCKS ---------------------------------------------------------------------------------------------------------

# SR LATCH NAND LOGIC --------------------------------------------------------------------------------------------------
class SR_LATCH:
    def __init__(self, logic_id=0):
        self.logic_id = logic_id
        self.state = 0

    def logic(self, S, R):
        if not (R or S):
            pass
        elif R:
            self.state = 0
        elif S:
            self.state = 1
        else:
            pass

        Q = self.state
        nQ = not Q

        return Q, nQ


# FLIP-FLOPS -----------------------------------------------------------------------------------------------------------
class D_FLIP_FLOP:
    """
    rising edge d-flip-flop implemented with NAND technology
    takes two rising edges to reach steady-state
    """

    def __init__(self, logic_id=0):
        self.logic_id = logic_id

        # initialize state
        self.clock_last_state = 0
        self.rising_edge = 0
        self.Q = 0

    def rising_edge_trigger(self, clock_state):
        if not self.clock_last_state and clock_state:
            self.rising_edge = True
        else:
            self.rising_edge = False

        self.clock_last_state = clock_state

    def logic_d_flip_flop(self, clk=1, d=0):
        self.rising_edge_trigger(clk)

        if self.rising_edge:
            if d:
                self.Q = 1
            else:
                self.Q = 0
        else:
            pass

        # for readability only
        Q = self.Q
        nQ = not Q

        return Q, nQ


# ADDER ----------------------------------------------------------------------------------------------------------------
def logic_full_adder(A, B, Cin):
    # sum
    sum = xor(Cin, xor(A, B))

    # carry out
    Cout = (A and B) or (B and Cin) or (A and Cin)

    return sum, Cout


# ACCUMULATOR BLOCK ----------------------------------------------------------------------------------------------------
class ONE_BIT_ACCUMULATOR:
    def __init__(self, logic_id=0):
        self.logic_id = logic_id
        self.dflipflop1 = D_FLIP_FLOP(1)
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
        s0, Cout0 = self.accumulator0.logic_one_bit_accumulator(clk=clk, y=y0, Cin=Cin)
        s1, Cout1 = self.accumulator1.logic_one_bit_accumulator(clk=clk, y=y1, Cin=Cout0)
        s2, Cout2 = self.accumulator2.logic_one_bit_accumulator(clk=clk, y=y2, Cin=Cout1)
        s3, Cout3 = self.accumulator3.logic_one_bit_accumulator(clk=clk, y=y3, Cin=Cout2)

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
