import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write,read
import soundfile as sf
fileName=""

def binary_to_decimal(binary_str):
    """
    Convert a two's complement binary string to a decimal integer.
    """
    if binary_str[0] == '1':  # negative number
        inverted_str = ''.join('1' if b == '0' else '0' for b in binary_str)
        return -1 * (int(inverted_str, 2) + 1)
    else:
        return int(binary_str, 2)

def decimal_to_binary(decimal, bits=32):
    """
    Convert a decimal integer to a two's complement binary string of 32 bits.
    """
    if decimal < 0:
        decimal = (1 << bits) + decimal
    binary_str = bin(decimal)[2:]  # convert to binary and remove '0b' prefix
    return binary_str.zfill(bits)

def twos_complement_addition(bin1, bin2, bits=32):
    """
    Perform two's complement binary addition with 32-bit padding.
    """
    # Ensure input binary strings are 32 bits
    bin1 = bin1.zfill(bits)
    bin2 = bin2.zfill(bits)

    dec1 = binary_to_decimal(bin1)
    dec2 = binary_to_decimal(bin2)
    result_decimal = dec1 + dec2
    
    # Wrap around if necessary
    max_value = 1 << (bits - 1)
    min_value = -1 * max_value
    if result_decimal >= max_value:
        result_decimal -= 2 * max_value
    elif result_decimal < min_value:
        result_decimal += 2 * max_value

    return decimal_to_binary(result_decimal, bits)

# Example usage
bin1 = "1111111111111111111111111111"  # -3 in 4-bit two's complement
bin2 = "00000000000000000000000000001"  # 6 in 4-bit two's complement
result = twos_complement_addition(bin1, bin2)
print(f"Result of adding {bin1} and {bin2} in 32-bit two's complement: {result}")
