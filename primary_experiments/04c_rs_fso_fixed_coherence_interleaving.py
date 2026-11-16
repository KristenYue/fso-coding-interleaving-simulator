# 04c_rs_fso_fixed_coherence_interleaving.py
# ============================================================
# TASK 04C:
# RS(255,223) over slow-fading FSO with GF(256) symbol interleaving
# Fixed coherence-length fading model
#
# 正式版：
#   Eb/N0 = 0:3:36 dB
#   MAX_FRAMES = 20000
#
# 核心改动：
#   慢衰落块长度固定为 COHERENCE_SYMBOLS = 8 个 RS symbols
#   不再让 fading block length = D
# ============================================================

import os
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt

try:
    import reedsolo
except ImportError:
    raise ImportError("请先安装 reedsolo：pip install reedsolo")


# ============================================================
# 参数区：正式版
# ============================================================

SEED = 12345
np.random.seed(SEED)
random.seed(SEED)

# RS(255,223)
RS_N = 255
RS_K = 223
RS_NSYM = RS_N - RS_K
RS_T = RS_NSYM // 2
RS_RATE = RS_K / RS_N

# 交织深度
INTERLEAVER_DEPTHS = [1, 8, 32, 128]

# 固定相干长度：重点参数
COHERENCE_SYMBOLS = 8

# 正式版 Eb/N0
EBN0_DB_LIST = list(range(0, 37, 3))

# 正式版停止条件
MIN_BIT_ERRORS = 200
MIN_FRAME_ERRORS = 100
MAX_FRAMES = 20000

# FSO 参数
RYTOV_VARIANCE = 0.75
ALPHA = 4.8685
BETA = 3.1649

# pointing 参数
POINTING_SIGMA = 0.100
A0 = 0.053972
XI = 1.5220

# 你之前用的是 h_p/A0，所以这里保持一致
USE_NORMALIZED_POINTING = True

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "04c_rs_fso_fixed_coherence_interleaving.csv")
PNG_PATH = os.path.join(RESULTS_DIR, "04c_rs_fso_fixed_coherence_interleaving.png")
PDF_PATH = os.path.join(RESULTS_DIR, "04c_rs_fso_fixed_coherence_interleaving.pdf")


# ============================================================
# RS 编码/译码
# ============================================================

rsc = reedsolo.RSCodec(RS_NSYM)


def rs_encode(msg_bytes: np.ndarray) -> np.ndarray:
    encoded = rsc.encode(bytes(msg_bytes))
    return np.frombuffer(encoded, dtype=np.uint8)


def rs_decode(codeword_bytes: np.ndarray):
    try:
        dec = rsc.decode(bytes(codeword_bytes))

        if isinstance(dec, tuple):
            decoded_msg = dec[0]
        else:
            decoded_msg = dec

        decoded_msg = np.frombuffer(bytes(decoded_msg), dtype=np.uint8)
        decoded_msg = decoded_msg[:RS_K]

        if len(decoded_msg) != RS_K:
            fallback = codeword_bytes[:RS_K].copy()
            return False, fallback

        return True, decoded_msg

    except reedsolo.ReedSolomonError:
        fallback = codeword_bytes[:RS_K].copy()
        return False, fallback


# ============================================================
# 工具函数
# ============================================================

def bytes_to_bits(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.astype(np.uint8))


def bits_to_bytes(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits.astype(np.uint8)).astype(np.uint8)


def count_bit_errors(a: np.ndarray, b: np.ndarray) -> int:
    xor = np.bitwise_xor(a.astype(np.uint8), b.astype(np.uint8))
    return int(np.unpackbits(xor).sum())


# ============================================================
# FSO fading
# ============================================================

def sample_gamma_gamma(num_samples: int) -> np.ndarray:
    x = np.random.gamma(shape=ALPHA, scale=1.0 / ALPHA, size=num_samples)
    y = np.random.gamma(shape=BETA, scale=1.0 / BETA, size=num_samples)
    return x * y


def sample_normalized_pointing(num_samples: int) -> np.ndarray:
    u = np.random.random(num_samples)
    return u ** (1.0 / (XI ** 2))


def sample_pointing(num_samples: int) -> np.ndarray:
    hp_norm = sample_normalized_pointing(num_samples)

    if USE_NORMALIZED_POINTING:
        return hp_norm
    else:
        return A0 * hp_norm


def sample_fso_fading(num_samples: int) -> np.ndarray:
    h_turb = sample_gamma_gamma(num_samples)
    h_point = sample_pointing(num_samples)
    return h_turb * h_point


def make_fixed_coherence_fading(num_symbols: int, coherence_symbols: int) -> np.ndarray:
    """
    给 interleaved 之后的 RS symbol 串生成慢衰落。

    重点：
        每 coherence_symbols 个 RS symbols 用同一个 fading。
        这里 coherence_symbols 固定为 8。
        它不等于 D。
    """
    num_blocks = int(math.ceil(num_symbols / coherence_symbols))
    h_blocks = sample_fso_fading(num_blocks)
    h_symbols = np.repeat(h_blocks, coherence_symbols)[:num_symbols]
    return h_symbols


# ============================================================
# OOK + AWGN + perfect CSI
# ============================================================

def ook_awgn_perfect_csi(tx_bits: np.ndarray, h_bits: np.ndarray, ebn0_db: float):
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)

    # 保持你之前的噪声缩放
    sigma2 = 1.0 / (4.0 * RS_RATE * ebn0_linear)
    sigma = math.sqrt(sigma2)

    x = tx_bits.astype(np.float64)

    noise = sigma * np.random.randn(len(tx_bits))
    y = h_bits * x + noise

    # perfect CSI threshold
    rx_bits = (y >= 0.5 * h_bits).astype(np.uint8)

    return rx_bits


# ============================================================
# GF(256) symbol interleaver
# ============================================================

def symbol_interleave(cw_matrix: np.ndarray) -> np.ndarray:
    """
    输入:
        cw_matrix shape = (D, 255)

    输出:
        tx_bytes length = D * 255

    顺序:
        cw0_sym0, cw1_sym0, ..., cwD-1_sym0,
        cw0_sym1, cw1_sym1, ..., cwD-1_sym1,
        ...
    """
    return cw_matrix.T.flatten()


def symbol_deinterleave(rx_stream: np.ndarray, D: int) -> np.ndarray:
    """
    symbol_interleave 的逆操作。

    输入:
        rx_stream length = D * 255

    输出:
        rx_cw_matrix shape = (D, 255)
    """
    return rx_stream.reshape(RS_N, D).T.copy()


# ============================================================
# 自检
# ============================================================

def run_self_tests():
    print("[1] Running self-tests...")

    msg = np.random.randint(0, 256, RS_K, dtype=np.uint8)
    cw = rs_encode(msg)
    ok, dec = rs_decode(cw)

    assert ok
    assert np.array_equal(msg, dec)
    print("RS self-test passed.")

    for D in INTERLEAVER_DEPTHS:
        x = np.random.randint(0, 256, size=(D, RS_N), dtype=np.uint8)
        s = symbol_interleave(x)
        y = symbol_deinterleave(s, D)
        assert np.array_equal(x, y)

    print("Symbol interleaver inverse self-test passed.")

    h = make_fixed_coherence_fading(255 * 32, COHERENCE_SYMBOLS)
    assert len(h) == 255 * 32
    print("Fixed-coherence fading self-test passed.")
    print()


# ============================================================
# 单个仿真点
# ============================================================

def simulate_one_point(D: int, ebn0_db: float):
    total_info_bits = 0
    total_coded_bits = 0

    raw_bit_errors = 0
    decoded_bit_errors = 0
    frame_errors = 0
    failures = 0
    miscorr = 0

    frames = 0
    start = time.time()

    while frames < MAX_FRAMES:
        # 一次 interleaver group 里面有 D 个 RS codewords
        frames += D

        # 1. 生成 D 个信息块
        msg_matrix = np.random.randint(0, 256, size=(D, RS_K), dtype=np.uint8)

        # 2. RS 编码
        cw_matrix = np.empty((D, RS_N), dtype=np.uint8)
        for d in range(D):
            cw_matrix[d, :] = rs_encode(msg_matrix[d, :])

        # 3. GF(256) symbol interleaving
        tx_bytes = symbol_interleave(cw_matrix)

        # 4. 固定相干长度慢衰落
        #    这里是关键：fading block length = COHERENCE_SYMBOLS，不是 D
        h_symbols = make_fixed_coherence_fading(
            num_symbols=len(tx_bytes),
            coherence_symbols=COHERENCE_SYMBOLS
        )

        # 5. byte 转 bit
        tx_bits = bytes_to_bits(tx_bytes)

        # 每个 RS symbol 是 1 byte，所以 8 个 bit 用同一个 fading
        h_bits = np.repeat(h_symbols, 8)

        # 6. 过信道
        rx_bits = ook_awgn_perfect_csi(tx_bits, h_bits, ebn0_db)

        # raw coded BER
        raw_bit_errors += int(np.sum(tx_bits != rx_bits))
        total_coded_bits += len(tx_bits)

        # 7. bit 转回 byte
        rx_bytes = bits_to_bytes(rx_bits)

        # 8. deinterleaving
        rx_cw_matrix = symbol_deinterleave(rx_bytes, D)

        # 9. RS 解码
        for d in range(D):
            success, decoded_msg = rs_decode(rx_cw_matrix[d, :])

            bit_err = count_bit_errors(decoded_msg, msg_matrix[d, :])
            decoded_bit_errors += bit_err

            if not success:
                failures += 1

            if bit_err > 0:
                frame_errors += 1

            if success and bit_err > 0:
                miscorr += 1

        total_info_bits += D * RS_K * 8

        if decoded_bit_errors >= MIN_BIT_ERRORS and frame_errors >= MIN_FRAME_ERRORS:
            break

    elapsed = time.time() - start

    raw_ber = raw_bit_errors / total_coded_bits

    if decoded_bit_errors > 0:
        ber = decoded_bit_errors / total_info_bits
        ber_upper = None
    else:
        ber = None
        ber_upper = 3.0 / total_info_bits

    if frame_errors > 0:
        fer = frame_errors / frames
        fer_upper = None
    else:
        fer = None
        fer_upper = 3.0 / frames

    return {
        "D": D,
        "EbN0_dB": ebn0_db,
        "raw_BER": raw_ber,
        "BER": ber,
        "BER_upper": ber_upper,
        "FER": fer,
        "FER_upper": fer_upper,
        "frames": frames,
        "failures": failures,
        "miscorr": miscorr,
        "time_s": elapsed,
        "decoded_bit_errors": decoded_bit_errors,
        "frame_errors": frame_errors,
        "total_info_bits": total_info_bits,
        "total_coded_bits": total_coded_bits,
    }


# ============================================================
# 保存 CSV
# ============================================================

def save_csv(results):
    header = [
        "D",
        "EbN0_dB",
        "raw_BER",
        "BER",
        "BER_upper",
        "FER",
        "FER_upper",
        "frames",
        "failures",
        "miscorr",
        "time_s",
        "decoded_bit_errors",
        "frame_errors",
        "total_info_bits",
        "total_coded_bits",
        "COHERENCE_SYMBOLS",
        "USE_NORMALIZED_POINTING",
    ]

    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")

        for r in results:
            row = [
                r["D"],
                r["EbN0_dB"],
                r["raw_BER"],
                "" if r["BER"] is None else r["BER"],
                "" if r["BER_upper"] is None else r["BER_upper"],
                "" if r["FER"] is None else r["FER"],
                "" if r["FER_upper"] is None else r["FER_upper"],
                r["frames"],
                r["failures"],
                r["miscorr"],
                r["time_s"],
                r["decoded_bit_errors"],
                r["frame_errors"],
                r["total_info_bits"],
                r["total_coded_bits"],
                COHERENCE_SYMBOLS,
                USE_NORMALIZED_POINTING,
            ]

            f.write(",".join(str(x) for x in row) + "\n")


# ============================================================
# 画图：BER
# ============================================================

def plot_ber(results):
    plt.figure(figsize=(9, 6))

    markers = {
        1: "o",
        8: "s",
        32: "^",
        128: "D",
    }

    for D in INTERLEAVER_DEPTHS:
        dres = [r for r in results if r["D"] == D]
        dres = sorted(dres, key=lambda x: x["EbN0_dB"])

        x = [r["EbN0_dB"] for r in dres]

        y = []
        upper_flags = []

        for r in dres:
            if r["BER"] is not None:
                y.append(r["BER"])
                upper_flags.append(False)
            else:
                y.append(r["BER_upper"])
                upper_flags.append(True)

        plt.semilogy(
            x,
            y,
            marker=markers.get(D, "o"),
            linewidth=1.5,
            label=f"D={D}"
        )

        xu = [xx for xx, up in zip(x, upper_flags) if up]
        yu = [yy for yy, up in zip(y, upper_flags) if up]

        if len(xu) > 0:
            plt.semilogy(
                xu,
                yu,
                linestyle="None",
                marker="v",
                markerfacecolor="none",
                markersize=8,
                label=f"D={D} upper"
            )

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Decoded BER")
    plt.title(
        "Task 04C: RS(255,223) GF(256) Symbol Interleaving over Slow-Fading FSO\n"
        f"Fixed coherence length = {COHERENCE_SYMBOLS} RS symbols, "
        "normalized pointing fluctuation, perfect CSI"
    )
    plt.ylim(1e-8, 1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(PNG_PATH, dpi=300)
    plt.savefig(PDF_PATH)
    plt.close()


# ============================================================
# 额外画图：FER
# ============================================================

def plot_fer(results):
    fer_png = os.path.join(RESULTS_DIR, "04c_rs_fso_fixed_coherence_interleaving_fer.png")
    fer_pdf = os.path.join(RESULTS_DIR, "04c_rs_fso_fixed_coherence_interleaving_fer.pdf")

    plt.figure(figsize=(9, 6))

    markers = {
        1: "o",
        8: "s",
        32: "^",
        128: "D",
    }

    for D in INTERLEAVER_DEPTHS:
        dres = [r for r in results if r["D"] == D]
        dres = sorted(dres, key=lambda x: x["EbN0_dB"])

        x = [r["EbN0_dB"] for r in dres]

        y = []
        upper_flags = []

        for r in dres:
            if r["FER"] is not None:
                y.append(r["FER"])
                upper_flags.append(False)
            else:
                y.append(r["FER_upper"])
                upper_flags.append(True)

        plt.semilogy(
            x,
            y,
            marker=markers.get(D, "o"),
            linewidth=1.5,
            label=f"D={D}"
        )

        xu = [xx for xx, up in zip(x, upper_flags) if up]
        yu = [yy for yy, up in zip(y, upper_flags) if up]

        if len(xu) > 0:
            plt.semilogy(
                xu,
                yu,
                linestyle="None",
                marker="v",
                markerfacecolor="none",
                markersize=8,
                label=f"D={D} upper"
            )

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("FER")
    plt.title(
        "Task 04C: RS(255,223) GF(256) Symbol Interleaving over Slow-Fading FSO\n"
        f"FER, fixed coherence length = {COHERENCE_SYMBOLS} RS symbols"
    )
    plt.ylim(1e-5, 1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fer_png, dpi=300)
    plt.savefig(fer_pdf)
    plt.close()


# ============================================================
# 主实验
# ============================================================

def run_experiment():
    print("=" * 90)
    print("TASK 04C: RS(255,223) OVER SLOW-FADING FSO WITH FIXED COHERENCE")
    print("=" * 90)
    print(f"RS parameters: n={RS_N}, k={RS_K}, nsym={RS_NSYM}, t={RS_T}")
    print(f"Code rate: R={RS_RATE:.6f}")
    print("Noise scaling: sigma^2 = 1 / (4 * R * Eb/N0)")
    print(f"Channel: Rytov variance={RYTOV_VARIANCE}, alpha={ALPHA:.4f}, beta={BETA:.4f}")
    print(f"Pointing: jitter sigma={POINTING_SIGMA:.3f} m, A0={A0:.6f}, xi={XI:.4f}")

    if USE_NORMALIZED_POINTING:
        print("Pointing gain used in simulation: h_p/A0")
    else:
        print("Pointing gain used in simulation: h_p")

    print("Interleaver type: GF(256) RS symbol-level")
    print(f"Fixed slow-fading coherence length: {COHERENCE_SYMBOLS} RS symbols")
    print("Important: fading block length is independent of interleaver depth D")
    print(f"Eb/N0 list: {EBN0_DB_LIST}")
    print(f"Interleaver depths: {INTERLEAVER_DEPTHS}")
    print(f"Minimum bit errors per point: {MIN_BIT_ERRORS}")
    print(f"Minimum frame errors per point: {MIN_FRAME_ERRORS}")
    print(f"Maximum frames per point: {MAX_FRAMES:,}")
    print()

    run_self_tests()

    print("[2] Running experiment...")
    print()

    all_results = []

    for D in INTERLEAVER_DEPTHS:
        print("-" * 90)
        print(f"Interleaver depth D={D}")
        print("-" * 90)

        for ebn0_db in EBN0_DB_LIST:
            res = simulate_one_point(D, ebn0_db)
            all_results.append(res)

            if res["BER"] is None:
                ber_text = f"BER upper={res['BER_upper']:.3e}"
            else:
                ber_text = f"BER={res['BER']:.3e}"

            if res["FER"] is None:
                fer_text = f"FER upper={res['FER_upper']:.3e}"
            else:
                fer_text = f"FER={res['FER']:.3e}"

            print(
                f"D={D:3d} | Eb/N0={ebn0_db:2.0f} dB | "
                f"raw BER={res['raw_BER']:.3e} | "
                f"{ber_text} | {fer_text} | "
                f"frames={res['frames']:,} | "
                f"failures={res['failures']} | "
                f"miscorr={res['miscorr']} | "
                f"time={res['time_s']:.1f}s"
            )

    save_csv(all_results)
    plot_ber(all_results)
    plot_fer(all_results)

    print()
    print("=" * 90)
    print("TASK 04C FINISHED")
    print("=" * 90)
    print(f"CSV saved to: {os.path.abspath(CSV_PATH)}")
    print(f"BER PNG saved to: {os.path.abspath(PNG_PATH)}")
    print(f"BER PDF saved to: {os.path.abspath(PDF_PATH)}")
    print(f"FER PNG saved to: {os.path.abspath(os.path.join(RESULTS_DIR, '04c_rs_fso_fixed_coherence_interleaving_fer.png'))}")
    print(f"FER PDF saved to: {os.path.abspath(os.path.join(RESULTS_DIR, '04c_rs_fso_fixed_coherence_interleaving_fer.pdf'))}")
    print("Open triangle markers represent zero-event 95% upper bounds.")
    print("This is the fixed-coherence RS symbol-level interleaving result.")


if __name__ == "__main__":
    run_experiment()