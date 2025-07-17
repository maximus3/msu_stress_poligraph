import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis


def load_data(file_path):
    """Загрузка данных из .npy файла"""
    data = np.load(file_path, allow_pickle=True).item()
    return data["signals"], data["labels"], data["sampling_rate"]


def normalize_data(signals):
    """Z-нормализация для каждого канала"""
    return [(channel - np.mean(channel)) / np.std(channel) for channel in signals]


def analyze_scr(signal, sr, peak_height=0.05, peak_prominence=0.03, min_distance=1.0):
    """Анализ SCR характеристик для одного канала с возвратом позиций пиков"""
    peaks, properties = find_peaks(
        signal,
        height=peak_height,
        prominence=peak_prominence,
        distance=int(sr * min_distance),
    )
    ns_scr = len(peaks)
    amp_scr = np.mean(properties["peak_heights"]) if ns_scr > 0 else 0.0

    recovery_times = []
    for i, peak in enumerate(peaks):
        half_amp = properties["peak_heights"][i] * 0.5
        recovery_signal = signal[peak:]
        cross_points = np.where(recovery_signal <= half_amp)[0]
        if len(cross_points) > 0:
            recovery_time = cross_points[0] / sr
            recovery_times.append(recovery_time)

    avg_recovery = np.mean(recovery_times) if recovery_times else 0.0
    return ns_scr, amp_scr, avg_recovery, peaks


def calculate_line_length(signal):
    """Вычисление длины линии как суммы абсолютных разностей между соседними точками"""
    return np.sum(np.abs(np.diff(signal)))


def plot_peaks(signal, peaks, filename, channel_name, sr, save_dir):
    """Сохранение графика с отмеченными пиками (только для SCR-каналов)"""
    plt.figure(figsize=(12, 4))
    time = np.arange(len(signal)) / sr

    plt.plot(time, signal, label="Сигнал")
    plt.scatter(time[peaks], signal[peaks], color="red", marker="x", label="Пики")

    plt.xlabel("Время, сек")
    plt.ylabel("Амплитуда")
    plt.title(f"Детекция пиков: {channel_name}")
    plt.legend()

    base_name = pathlib.Path(filename).stem
    safe_channel = channel_name.replace(" ", "_")
    plot_path = pathlib.Path(save_dir) / f"{base_name}_{safe_channel}_peaks.png"

    plt.savefig(str(plot_path), bbox_inches="tight")
    plt.close()


def process_file(file_path, start_sec, end_sec, selected_channels, save_dir):
    """Обработка одного файла с сохранением графиков только для SCR-каналов"""
    try:
        data = np.load(file_path, allow_pickle=True).item()
        signals = np.array(data["signals"])
        labels = data["labels"]
        sr = data["sampling_rate"]

        # Обрезка временного интервала
        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)
        if end_idx > signals.shape[1]:
            raise ValueError("Интервал анализа превышает длину сигнала")
        signals = signals[:, start_idx:end_idx]

        # Выбор каналов
        if selected_channels != "all":
            channel_indices = [
                i for i, lbl in enumerate(labels) if lbl in selected_channels
            ]
            signals = signals[channel_indices]
            labels = [labels[i] for i in channel_indices]

        # Нормализация
        normalized = normalize_data(signals)

        # Обработка для каждого канала
        results = []
        for i, (chan, label) in enumerate(zip(normalized, labels)):
            original_signal = signals[i]
            line_length = calculate_line_length(chan)

            # Определяем тип канала
            is_scr_channel = label in ["scr l", "scr r"]

            if is_scr_channel:
                # Полный анализ для SCR-каналов
                ns, amp, rt, peaks = analyze_scr(chan, sr)

                # Дополнительные статистики
                raw_sd = np.std(original_signal, ddof=1)
                norm_sd = np.std(chan, ddof=1)
                diff_signal = np.diff(chan)
                rmssd = np.sqrt(np.mean(diff_signal**2))
                sk = skew(chan)
                kurt_val = kurtosis(chan)
                mean_val = np.mean(original_signal)
                fano = (
                    np.var(original_signal, ddof=1) / mean_val
                    if abs(mean_val) > 1e-7
                    else np.nan
                )

                # Сохранение графика пиков
                base_name = pathlib.Path(file_path).name.replace("_processed.npy", "")
                plot_peaks(chan, peaks, base_name, label, sr, save_dir)
            else:
                # Только длина линии для не-SCR каналов
                ns, amp, rt = np.nan, np.nan, np.nan
                raw_sd, norm_sd, rmssd, sk, kurt_val, fano = [np.nan] * 6

            # Сохранение результатов
            results.append(
                {
                    "Channel": label,
                    "NS-SCR": ns,
                    "Amp-SCR": amp,
                    "Recovery-Time": rt,
                    "Line-Length": line_length,
                    "Raw-SD": raw_sd,
                    "Norm-SD": norm_sd,
                    "RMSSD": rmssd,
                    "Skewness": sk,
                    "Kurtosis": kurt_val,
                    "Fano-Factor": fano,
                }
            )

        return results
    except Exception as e:
        print(f"Ошибка обработки {file_path}: {str(e)}")
        return None


def main():
    # Настройки
    data_dir = pathlib.Path("data/result/")
    start_sec = 15
    end_sec = 315
    selected_channels = [
        "scr l",
        "scr r",  # SCR-каналы (полный анализ + длина линии)
        "pneumogram h",
        "pneumogram l",  # Другие каналы (только длина линии)
        "HR (calculated)",
        "ppg l",
        "ppg r",
    ]

    all_results = []
    for file_path in data_dir.glob("*_processed.npy"):
        metrics = process_file(
            file_path, start_sec, end_sec, selected_channels, data_dir
        )

        if metrics:
            for chan_metrics in metrics:
                all_results.append(
                    {
                        "File": file_path.name.replace("_processed.npy", ""),
                        **chan_metrics,
                    }
                )

    # Сохранение результатов
    df = pd.DataFrame(all_results)
    output_path = data_dir / "SCR_analysis_results.csv"
    df.to_csv(str(output_path), index=False, sep=";", decimal=",", encoding="utf-8-sig")

    print(f"\nРезультаты анализа ({len(df)} записей):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
