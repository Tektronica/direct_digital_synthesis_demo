import numpy as np
import matplotlib.pyplot as plt


# AD9833 Theory of Operation (11,12) https://www.analog.com/media/en/technical-documentation/data-sheets/ad9833.pdf

def run_demo():
    start()


def start():
    log = {"time": [], "phase": [], "address": [], "output": []}
    M = 16
    N = 8
    dac_bit_depth = 8

    fc = 16e6
    time = 0  # ns
    tf = 2000  # ns
    dt = (1 / fc) * 1e9  # time step in nanoseconds

    NCO = NUMERICALLY_CONTROLLED_OSCILLATOR(M=M, N=N, fc=fc)

    while time < tf:
        last_phase, rom_address = NCO.phase_accumulator()
        dac_code = sin_ROM(rom_address, dac_bit_depth)
        output = dac(dac_code, dac_bit_depth)

        # log
        log["time"].append(time)
        log["phase"].append(last_phase)
        log["address"].append(rom_address)
        log["output"].append(output)

        # increment time
        time += dt

    plot(log)


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


def low_pass():
    pass


class NUMERICALLY_CONTROLLED_OSCILLATOR:
    def __init__(self, M, N, fc):
        self.fc = fc  # clock frequency
        self.N = N  # bit depth of the accumulator
        self.phase_register = 0

        # set amount the phase accumulator is incremented each clock cycle
        self.M = 0
        self.set_frequency_tuning_word(M)

    # CONVENIENCE FUNCTIONS --------------------------------------------------------------------------------------------
    def get_tuning(self):
        return (self.M * self.fc) / (2 ** self.N)

    def get_frequency_resolution(self):
        # The resolution of the DAC is typically 2 to 4 bits less than the width of the lookup table.
        return self.fc / (2 ** self.N)

    # FREQUENCY TUNING WORD --------------------------------------------------------------------------------------------
    def set_frequency_tuning_word(self, new_M):
        # set the frequency tuning word, M
        # modulo-M counter incrementing a stored number each time it receives a clock pulse
        self.M = new_M

    def get_frequency_tuning_word(self):
        # retrieve the frequency tuning word, M
        # modulo-M counter incrementing a stored number each time it receives a clock pulse
        return self.M

    # PHASE REGISTER ---------------------------------------------------------------------------------------------------
    def delta_phase(self):
        # 0 < ΔPhase < (2^N − 1)
        pass

    def set_phase_register(self, last_phase):
        # N bit register, which is loaded the modulus 2^N sum of its old output and the frequency tuning word
        self.phase_register = last_phase

    def get_phase_register(self):
        # N bit register, which is loaded the modulus 2^N sum of its old output and the frequency tuning word
        return self.phase_register

    def reset_phase_register(self):
        # N bit register, which is loaded the modulus 2^N sum of its old output and the frequency tuning word
        self.phase_register = 0

    # PHASE ACCUMULATOR ------------------------------------------------------------------------------------------------
    def phase_accumulator(self):
        # modulo-M counter incrementing a stored number each time it receives a clock pulse
        # returns truncated output as an address to the lookup table

        M = self.get_frequency_tuning_word()  # M is retrieved from the delta phase register
        last_phase = self.get_phase_register()  # last phase retrieved from the phase accumulator register

        # integrate the frequency tuning word (Phase is the integral of frequency)
        if last_phase < (2 ** self.N - M):
            new_phase = (last_phase + M)
            self.set_phase_register(new_phase)
        else:
            self.reset_phase_register()  # resets the value of the phase register

        ram_address = np.mod(last_phase, 2 ** self.N)

        return last_phase, ram_address


def plot(log):
    # log
    time = log["time"]
    phase = log["phase"]
    address = log["address"]
    output = log["output"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle('Direct Digital Synthesis (DDS)', fontsize=18)

    ax1.plot(time, phase)
    ax1.set_title('phase')
    ax1.set_xlabel('barrel length (travelled) (cm)')
    ax1.set_ylabel('phase')
    ax1.grid()

    ax2.plot(time, output)
    ax2.set_title('output')
    ax2.set_xlabel('time (ns)')
    ax2.set_ylabel('dac output')
    ax2.grid()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_demo()
