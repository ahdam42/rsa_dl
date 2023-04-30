import tensorflow as tf
from Crypto.Util import number

BIT_NUMBER = 896
BATCH_SIZE = 16384
RSA_NUM = 412023436986659543855531365332575948179811699844327982845455626433876445565248426198098870423161841879261420247188869492560931776375033421130982397485150944909106910269861031862704114880866970564902903653658867433731720813104105190864254793282601391257624033946373269391

def vec_to_int(bin_array):
    int_ = 0
    for bit in bin_array:
        int_ = (int_ << 1) | int(bit > 0.5)
    return int_

def int_to_vec(n):
    return tf.convert_to_tensor([int(digit) for digit in bin(n)[2:]])

def generate_rsa_pairs():
    p = number.getPrime(BIT_NUMBER // 2)
    q = number.getPrime(BIT_NUMBER // 2)
    n = int_to_vec(p * q)
    if len(n) == BIT_NUMBER:
        return n, int_to_vec(min(p, q))
    else:
        return generate_rsa_pairs()

RSA_NUM_VEC = int_to_vec(RSA_NUM)