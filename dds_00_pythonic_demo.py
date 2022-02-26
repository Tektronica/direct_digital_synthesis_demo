import numpy as np
import matplotlib.pyplot as plt
import spectrum_analyzer as sa
from scipy import signal

"""
# AD9833 Theory of Operation (11,12)
https://www.analog.com/media/en/technical-documentation/data-sheets/ad9833.pdf

# All About Direct Digital Synthesis
https://www.analog.com/en/analog-dialogue/articles/all-about-direct-digital-synthesis.html
https://www.analog.com/media/en/analog-dialogue/volume-38/number-3/articles/all-about-direct-digital-synthesis.pdf

An Accurate DDS Method Using Compound Frequency Tuning Word and Its FPGA Implementation
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwir5fzEqJz2AhUrGDQIHc7gAdcQFnoECAkQAQ&url=https%3A%2F%2Fwww.mdpi.com%2F2079-9292%2F7%2F11%2F330%2Fpdf-vor&usg=AOvVaw291okQKs9_yy0JqMenWPAb

Calculating the DDS tuning word (floating point)
https://jimhollister.wordpress.com/2015/09/07/calculating-the-dds-tuning-word/
"""


def run_demo():
    start()


def start():
    data = {
        "time": [], "phase": [], "address": [], "output": [],
        "filtered": [], "xf": [], "yf": [], "xf_f": [], "yf_f": []
    }

    fs = 50e6  # sampling frequency of simulation

    fc = 1e6  # output frequency
    fmax = 2e6  # maximum output frequency used for filter cutoff

    fosc = 16e6  # clock frequency of the DDS
    N = 8  # N-bit phase accumulator
    dac_bit_depth = 8  # N-bit DAC

    # TIME STEP AND RUN TIME -------------------------------------------------------------------------------------------
    time = 0.0  # ns
    tf = 1e7  # ns
    dt = (1 / fs) * 1e9  # time step in nanoseconds

    # RUN SIMULATION ===================================================================================================
    last_phase = 0
    rom_address = 0
    output = 0.0

    NCO = NUMERICALLY_CONTROLLED_OSCILLATOR(N=N, fosc=fosc)
    NCO.set_output_frequency(fc)

    RO = ROLLOVER()  # initialize instance of Rollover class to synchronize sampling and clock frequencies

    while time < tf:
        if RO.check_rollover(time, ((1 / fosc) * 1e9)):
            last_phase, rom_address = NCO.phase_accumulator()
            dac_code = sin_ROM(rom_address, dac_bit_depth)
            output = dac(dac_code, dac_bit_depth)

        # log
        data["time"].append(time)
        data["phase"].append(last_phase)
        data["address"].append(rom_address)
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


class NUMERICALLY_CONTROLLED_OSCILLATOR:
    def __init__(self, N, fosc):
        self.fosc = fosc  # clock frequency
        self.N = N  # bit depth of the accumulator
        self.phase_register = 0
        self.M = 1  # tuning bit the phase accumulator is incremented by on each clock cycle

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
        M = self.get_frequency_tuning_word()  # M is retrieved from the delta phase register
        last_phase = self.get_phase_register()  # last phase retrieved from the phase accumulator register

        # integrate the frequency tuning word (phase is the integral of frequency)
        ram_address = (last_phase + M) % (2**self.N)
        self.set_phase_register(ram_address)

        return last_phase, ram_address


def plot(data, xt_limits, xf_limits, M=0):
    # log
    time = data["time"]
    phase = data["phase"]
    address = data["address"]
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
