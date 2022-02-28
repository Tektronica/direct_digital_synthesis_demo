import numpy as np
import matplotlib.pyplot as plt
import spectrum_analyzer as sa
from scipy import signal

"""
DDS demo with 4-bit accumulator
"""


def run_demo():
    start()


# def start():
#     print('D Flip Flop test')
#     dff = D_FLIP_FLOP()
#     dff.logic_d_flip_flop(clk=0, d=1)
#     print(2, 'rising edge ---->', dff.logic_d_flip_flop(clk=1, d=1), 'should change to 1')
#     dff.logic_d_flip_flop(clk=0, d=0)
#     print(4, 'rising edge ---->', dff.logic_d_flip_flop(clk=1, d=0), 'change to 0')
#     dff.logic_d_flip_flop(clk=0, d=1)
#     print(6, 'rising edge ---->', dff.logic_d_flip_flop(clk=1, d=1), 'change to 1')
#
#     print('\nfull adder test')
#     print('0+0 with Cin=0 --> sum = 0, Cout = 0 | ', logic_full_adder(A=0, B=0, Cin=0))
#     print('0+0 with Cin=1 --> sum = 1, Cout = 0 | ', logic_full_adder(A=0, B=0, Cin=1))
#     print('0+1 with Cin=0 --> sum = 1, Cout = 0 | ', logic_full_adder(A=0, B=1, Cin=0))
#     print('0+1 with Cin=1 --> sum = 0, Cout = 1 | ', logic_full_adder(A=0, B=1, Cin=1))
#     print('1+0 with Cin=0 --> sum = 1, Cout = 0 | ', logic_full_adder(A=1, B=0, Cin=0))
#     print('1+0 with Cin=1 --> sum = 0, Cout = 1 | ', logic_full_adder(A=1, B=0, Cin=1))
#     print('1+1 with Cin=0 --> sum = 0, Cout = 1 | ', logic_full_adder(A=1, B=1, Cin=0))
#     print('1+1 with Cin=1 --> sum = 1, Cout = 1 | ', logic_full_adder(A=1, B=1, Cin=1))
#
#     print('\nfour bit accumulator')
#     FBA = FOUR_BIT_ACCUMULATOR()
#     for i in range(1, 16):
#         binary_sum = f'0b{i:04b}'[2:]
#         print(f'--> [{i}] sum = {binary_sum} | actual:', FBA.four_bit_accumulator(clk=1, y='0001', Cin=0))
#         FBA.four_bit_accumulator(clk=0, y='0000', Cin=0)

def start():
    data = {
        "time": [], "phase": [], "output": [], "filtered": [],
        "xf": [], "yf": [], "xf_f": [], "yf_f": []
    }

    fs = 50e6  # sampling frequency of simulation

    fc = 1e6  # output frequency
    fmax = 2e6  # maximum output frequency used for filter cutoff

    fosc = 16e6  # clock frequency of the DDS
    N = 4  # N-bit phase accumulator
    dac_bit_depth = 4  # N-bit DAC

    # TIME STEP AND RUN TIME -------------------------------------------------------------------------------------------
    time = 0.0  # ns
    tf = 1e7  # ns
    dt = (1 / fs) * 1e9  # time step in nanoseconds

    # RUN SIMULATION ===================================================================================================
    phase_address = 0
    output = 0.0

    NCO = NUMERICALLY_CONTROLLED_OSCILLATOR(N=N, fosc=fosc)
    NCO.set_output_frequency(fc)

    RO = ROLLOVER()  # initialize instance of Rollover class to synchronize sampling and clock frequencies

    while time < tf:
        if RO.check_rollover(time, ((1 / fosc) * 1e9)):
            phase_address = NCO.phase_accumulator()  # returns the last stored phase address
            dac_code = sin_ROM(phase_address, dac_bit_depth)  # returns the dac code from lookup table
            output = dac(dac_code, dac_bit_depth)  # returns the dac output value

        # log
        data["time"].append(time)
        data["phase"].append(phase_address)
        data["output"].append(output)

        # increment time
        time += dt

    # FILTER OUTPUT ----------------------------------------------------------------------------------------------------
    # sampling frequency is twice the clock of the system to see the impact on the spectrum
    data["filtered"] = low_pass(data["output"], fmax=fmax, fs=fs)

    # ANALYZE SPECTRUM -------------------------------------------------------------------------------------------------
    yt = data["output"]
    yt_f = data["filtered"]

    xf, yf, xf_real, yf_real, mlw = sa.windowed_fft(yt=yt, Fs=fs, N=len(yt), windfunc='blackman')
    data["xf"] = xf_real / 1e6
    data["yf"] = 20 * np.log10(np.abs(yf_real / max(abs(yf_real))))

    xf_f, yf_f, xf_real_f, yf_real_f, mlw_f = sa.windowed_fft(yt=yt_f, Fs=fs, N=len(yt_f), windfunc='blackman')
    data["xf_f"] = xf_real_f / 1e6
    data["yf_f"] = 20 * np.log10(np.abs(yf_real_f / max(abs(yf_real_f))))  # TODO: normalize to the unfiltered output??

    # COMPUTE LIMITS ---------------------------------------------------------------------------------------------------
    xt_limits = (0, (4 / fc) * 1e9)

    # set max to not exceed max bin
    xf_limits = (min(xf_real) / 1e6, min(10 ** (np.ceil(np.log10(fc)) + 1), fs / 2 - fs / N) / 1e6)

    # PLOT -------------------------------------------------------------------------------------------------------------
    plot(data, xt_limits, xf_limits, M=NCO.get_frequency_tuning_word())


class ROLLOVER:
    def __init__(self):
        self.last_remainder = 0

    def check_rollover(self, a, n):
        """
        True if rollover using modulo operation
        " a mod n "

        :param a:
        :param n:
        :return: rollover
        """
        this_remainder = a % n
        result = this_remainder <= self.last_remainder
        self.last_remainder = this_remainder

        return result


def sin_ROM(address, dac_bit_depth):
    """
    converts phase information into amplitude
    + sin_rom emulated by passing phase into function to match look-up table
    + lookup table must have 2^N entries to match the size of the accumulator

    :return: dac_code
    """
    bit_depth = 2 ** dac_bit_depth
    dac_code = np.sin(2 * np.pi * address / bit_depth) * (2 ** dac_bit_depth)

    return dac_code


def dac(dac_code, dac_bit_depth):
    N = dac_bit_depth
    return 3.3 * dac_code / (2 ** N - 1)


def low_pass(sig, fmax, fs):
    """
    Digital high-pass filter at 15 Hz to remove the 10 Hz tone

    :return: filtered output
    """
    order = 10  # filter order

    # cutoff should be below (not equal to) nyquist of the sampling frequency
    cutoff = fmax

    # since fs is specified, we do not normalize the cutoff to the nyquist (fc/fnyq
    sos = signal.butter(N=order, Wn=cutoff, btype='lowpass', analog=False, output='sos', fs=fs)
    filtered = signal.sosfilt(sos, sig)

    return filtered


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
        y3 = int(y[0])  # LSB
        y2 = int(y[1])
        y1 = int(y[2])
        y0 = int(y[3])  # MSB

        # register of flip-flops
        s0, Cout0 = self.accumulator0.logic_one_bit_accumulator(clk=clk, y=y0, Cin=Cin)
        s1, Cout1 = self.accumulator1.logic_one_bit_accumulator(clk=clk, y=y1, Cin=Cout0)
        s2, Cout2 = self.accumulator2.logic_one_bit_accumulator(clk=clk, y=y2, Cin=Cout1)
        s3, Cout3 = self.accumulator3.logic_one_bit_accumulator(clk=clk, y=y3, Cin=Cout2)

        return str(s3) + str(s2) + str(s1) + str(s0), Cout3


class NUMERICALLY_CONTROLLED_OSCILLATOR:
    def __init__(self, N, fosc):
        self.fosc = fosc  # clock frequency
        self.N = N  # bit depth of the accumulator
        self.phase_register = 0
        self.M = 1  # tuning bit the phase accumulator is incremented by on each clock cycle
        self.FBA = FOUR_BIT_ACCUMULATOR()

    # CONVENIENCE FUNCTIONS ============================================================================================
    def get_tuning(self):
        """
        the tuning word, M, adjusts the frequency of output by varying the phase accumulator incrementation on each
        clock cycle.

        :return: tuning frequency
        """
        return (self.M * self.fosc) / (2 ** self.N)

    def get_frequency_resolution(self):
        """
        The resolution of the DAC is typically 2 to 4 bits less than the width of the lookup table.
        +   in practical DDS systems, 13 to 15 MSBs are truncated to reduce the lookup table size without affecting
            resolution. Impacts phase noise.

        :return: frequency resolution
        """
        return self.fosc / (2 ** self.N)

    # FREQUENCY TUNING WORD ============================================================================================
    def set_output_frequency(self, fc):
        """
        Calculates the tuning word, M for a given output frequency as a function of the sampling frequency
        +   Since the accumulator bit width is fixed, the frequency tuning word cannot be decimal.
        +   Output frequency cannot be arbitrary value.

        param fc: output frequency
        """
        new_M = int((2 ** self.N) * (fc / self.fosc))
        self.set_frequency_tuning_word(new_M)

    def set_frequency_tuning_word(self, new_M):
        # set the frequency tuning word, M
        # modulo-M counter incrementing a stored number each time it receives a clock pulse
        self.M = new_M

    def get_frequency_tuning_word(self):
        # retrieve the frequency tuning word, M
        # modulo-M counter incrementing a stored number each time it receives a clock pulse
        return self.M

    # PHASE REGISTER ===================================================================================================
    def set_phase_register(self, last_phase):
        # N bit register, which is loaded the modulus 2^N sum of its old output and the frequency tuning word
        self.phase_register = last_phase

    def get_phase_register(self):
        # N bit register, which is loaded the modulus 2^N sum of its old output and the frequency tuning word
        return self.phase_register

    def reset_phase_register(self):
        # N bit register, which is loaded the modulus 2^N sum of its old output and the frequency tuning word
        self.phase_register = 0

    # PHASE ACCUMULATOR ================================================================================================
    def phase_accumulator(self):
        """
        modulo-M counter incrementing a stored number each time it receives a clock pulse
        returns truncated output as an address to the lookup table

        An N-bit phase accumulator requires a 2^N lookup table.
        If each entry is stored with k-bit accuracy, then the lookup table size in memory is (k*2^N)/8e9 [Gigabytes]
        """
        M = f'0b{self.get_frequency_tuning_word():04b}'[2:]  # M is retrieved from the delta phase register
        last_phase_address = self.get_phase_register()  # last phase retrieved from the phase accumulator register

        # integrate the frequency tuning word (phase is the integral of frequency)
        next_address, carry = self.FBA.four_bit_accumulator(clk=1, y=M, Cin=0)  # "tick"
        self.FBA.four_bit_accumulator(clk=0, y='0000', Cin=0)  # "tock"
        self.set_phase_register(int(next_address, 2))

        return last_phase_address


def plot(data, xt_limits, xf_limits, M=0):
    # log
    time = data["time"]
    phase = data["phase"]
    output = data["output"]
    filtered = data["filtered"]

    xf = data["xf"]
    yf = data["yf"]
    yf_f = data["yf_f"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
    fig.suptitle('Direct Digital Synthesis (DDS)', fontsize=18)

    # phase accumulator
    ax1.step(time, phase)
    ax1.set_xlim(xt_limits)
    ax1.set_title(f'Phase Accumulator ($0 \leq Phase < 2^N$) where tuning word = {M} (M)')
    ax1.set_xlabel('time (ns)')
    ax1.set_ylabel('Phase Increment')
    ax1.grid()

    # dac output / filtered output
    ax2.step(time, output)
    ax2.plot(time, filtered, color='#C02942', linestyle='--')
    ax2.set_xlim(xt_limits)
    ax2.set_title('DAC Output / Filtered Output')
    ax2.set_xlabel('time (ns)')
    ax2.set_ylabel('Output')
    ax2.legend(['DAC Output', 'Filtered Output'])
    ax2.grid()

    # spectrum
    ax3.plot(xf, yf)
    ax3.plot(xf, yf_f, color='#C02942', linestyle='--')
    ax3.set_xlim(xf_limits)
    ax3.set_title('Normalized Spectrum')
    ax3.set_xlabel('Frequency (MHz)')
    ax3.set_ylabel('Magnitude')
    ax3.legend(['DAC Output', 'Filtered Output'])
    ax3.grid()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_demo()
