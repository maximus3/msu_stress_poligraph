import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pathlib
import math

# Пути и настройки
path_bdf = pathlib.Path("data/raw/")
path_to_save = pathlib.Path("data/result/")
samp_freq = 100  # Целевая частота дискретизации

# Параметры временного интервала для визуализации (ДОБАВЛЕНО)
start_time_sec = 15.0  # Начало интервала визуализации (сек)
end_time_sec = 315.0  # Конец интервала визуализации (сек)

# Параметры фильтров для каналов: [Верх.дых, Ниж.дых, КГР, ФПГ, ЧСС]
orderLPF = [1, 1, 1, 2, 0]
orderHPF = [1, 1, 1, 2, 0]
HPF = [0.1, 0.1, 0.1, 1.25, 0.0]
LPF = [0.2, 0.2, 0.25, 12.5, 0.0]


def lowpassfilter(sig, s_freq, order, cutoff_freq):
    """Функция для низкочастотной фильтрации"""
    if order == 0 or cutoff_freq == 0:
        return sig
    sos = signal.butter(order, cutoff_freq, "lp", fs=s_freq, output="sos")
    return signal.sosfilt(sos, sig)


def highpassfilter(sig, s_freq, order, cutoff_freq):
    """Функция для высокочастотной фильтрации"""
    if order == 0 or cutoff_freq == 0:
        return sig
    sos = signal.butter(order, cutoff_freq, "hp", fs=s_freq, output="sos")
    return signal.sosfilt(sos, sig)


def calculate_hr_from_ppg(ppg_signal, fs):
    """Расчет ЧСС из сигнала ФПГ с использованием CSS-метода (улучшенная версия)"""
    # 1. Параметры фильтрации для CSS (оптимизированные)
    SamplePeriodForFilters = 1.0 / fs
    theApp_LPF_CSS1 = 2.0  # Hz
    theApp_HPF_CSS1 = 0.5  # Hz (уменьшено для лучшего подавления базальной линии)
    theApp_HPF_CSS2 = 2.0  # Hz

    # 2. Расчет констант фильтрации
    LPFConst_CSS1 = 1.0 - math.exp(-6.28 * theApp_LPF_CSS1 * SamplePeriodForFilters)
    HPFConst_CSS1 = 1.0 - math.exp(-6.28 * theApp_HPF_CSS1 * SamplePeriodForFilters)
    HPFConst_CSS2 = 1.0 - math.exp(-6.28 * theApp_HPF_CSS2 * SamplePeriodForFilters)

    # 3. Инициализация переменных фильтров
    LPF_Voltage_CSS1 = ppg_signal[0] if len(ppg_signal) > 0 else 0.0
    HPF_Voltage_CSS1 = 0.0
    HPF_Voltage_CSS2 = 0.0
    CSS_Prefiltered = np.zeros(len(ppg_signal))

    # 4. Фильтрация сигнала ФПГ (улучшенная стабильность)
    for sample in range(len(ppg_signal)):
        current_sample = ppg_signal[sample]

        # Первый LPF
        LPF_Voltage_CSS1 += (current_sample - LPF_Voltage_CSS1) * LPFConst_CSS1

        # Первый HPF
        HPF_Voltage_CSS1 += (LPF_Voltage_CSS1 - HPF_Voltage_CSS1) * HPFConst_CSS1
        CSS_Prefiltered[sample] = LPF_Voltage_CSS1 - HPF_Voltage_CSS1

        # Второй HPF
        HPF_Voltage_CSS2 += (CSS_Prefiltered[sample] - HPF_Voltage_CSS2) * HPFConst_CSS2
        CSS_Prefiltered[sample] = CSS_Prefiltered[sample] - HPF_Voltage_CSS2

    # 5. Адаптивное определение порога
    # Используем скользящее окно вместо глобального std
    window_size = 5 * fs  # 5 секунд
    threshold_arr = np.zeros(len(ppg_signal))

    for i in range(len(ppg_signal)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(ppg_signal), i + window_size // 2)
        window = CSS_Prefiltered[start_idx:end_idx]

        if len(window) > 0:
            # Берем 70-й перцентиль вместо std
            threshold_arr[i] = np.percentile(np.abs(window), 70) * 0.7
        else:
            threshold_arr[i] = 0.01  # минимальное значение по умолчанию

    # 6. Расчет ЧСС с улучшенной логикой
    hr_signal = np.zeros(len(ppg_signal))
    CSS_Time = 0.0
    CSS_Value = 60.0  # начальное значение ЧСС
    CSS_Value_prev = 60.0
    CSS_Prefiltered_Prev = CSS_Prefiltered[0] if len(ppg_signal) > 0 else 0.0
    last_peak_time = 0.0
    min_interval = 0.33  # ~180 уд/мин
    max_interval = 2.0  # ~30 уд/мин

    for sample in range(1, len(ppg_signal)):
        CSS_Time += SamplePeriodForFilters

        # Детекция фронта (пересечение порога снизу вверх)
        if (
            CSS_Prefiltered[sample] > threshold_arr[sample]
            and CSS_Prefiltered_Prev <= threshold_arr[sample - 1]
        ):
            if CSS_Time > min_interval and CSS_Time < max_interval:
                # Рассчитываем мгновенную ЧСС
                instant_hr = 60.0 / CSS_Time

                # Плавное изменение ЧСС (фильтр первого порядка)
                CSS_Value = 0.2 * instant_hr + 0.8 * CSS_Value_prev
                CSS_Value_prev = CSS_Value
                last_peak_time = sample * SamplePeriodForFilters

            CSS_Time = 0.0

        CSS_Prefiltered_Prev = CSS_Prefiltered[sample]

        # Если давно не было пиков - плавно уменьшаем ЧСС
        if (sample * SamplePeriodForFilters - last_peak_time) > 2.0:
            CSS_Value = CSS_Value_prev * 0.95

        hr_signal[sample] = CSS_Value

    return hr_signal


# Создаем папку для результатов
path_to_save.mkdir(exist_ok=True)

# Получаем список файлов
file_names = [f.stem for f in path_bdf.glob("*.bdf")]

print(f"Files to process in {path_bdf}:")
for name in file_names:
    print(f" - {name}.bdf")

# Обработка файлов
for file_name in file_names:
    print(f"\nProcessing {file_name}...")

    with pyedflib.EdfReader(str(path_bdf / f"{file_name}.bdf")) as f:
        channels = f.getSignalLabels()
        processed_signals = []
        ch_new = []
        ppg_signal = None
        ppg_index = None

        # Обрабатываем каждый канал
        for ch_num in range(f.signals_in_file):
            original_signal = f.readSignal(ch_num)
            original_rate = f.getSampleFrequency(ch_num)

            # Ресемплинг
            resampled = signal.resample(
                original_signal, int(len(original_signal) * samp_freq / original_rate)
            )

            # Определяем тип канала по метке
            label = channels[ch_num].lower()

            # Применяем соответствующие фильтры
            if "pneumogram h" in label:
                channel_type = 0
            elif "pneumogram l" in label:
                channel_type = 1
            elif "scr" in label or "gsr" in label:
                channel_type = 2
            elif "ppg" in label:
                channel_type = 3
                # Сохраняем ФПГ для расчета ЧСС
                ppg_signal = resampled.copy()
                ppg_index = ch_num
            elif "hr" in label or "heart" in label:
                channel_type = 4
            else:
                continue

            # Применяем фильтрацию
            filtered = highpassfilter(
                resampled, samp_freq, orderHPF[channel_type], HPF[channel_type]
            )
            filtered = lowpassfilter(
                filtered, samp_freq, orderLPF[channel_type], LPF[channel_type]
            )

            # Инверсия сигнала при необходимости
            if channel_type in [0, 1, 2, 3]:
                filtered *= -1

            processed_signals.append(filtered)
            ch_new.append(label)

        # Расчет ЧСС из ФПГ при наличии
        if ppg_signal is not None:
            print("Calculating HR from PPG...")
            # Применяем базовую фильтрацию к ФПГ
            ppg_filtered = highpassfilter(ppg_signal, samp_freq, orderHPF[3], HPF[3])
            ppg_filtered = lowpassfilter(ppg_filtered, samp_freq, orderLPF[3], LPF[3])
            ppg_filtered *= -1

            # Рассчитываем ЧСС
            hr_calculated = calculate_hr_from_ppg(ppg_filtered, samp_freq)

            # Заменяем или добавляем канал ЧСС
            hr_exists = any(
                "hr" in label.lower() or "heart" in label.lower() for label in ch_new
            )

            if hr_exists:
                # Заменяем существующий канал ЧСС
                for i, label in enumerate(ch_new):
                    if "hr" in label.lower() or "heart" in label.lower():
                        processed_signals[i] = hr_calculated
                        ch_new[i] = "HR (calculated)"
                        print("Replaced existing HR channel with calculated HR")
                        break
            else:
                # Добавляем новый канал
                processed_signals.append(hr_calculated)
                ch_new.append("HR (calculated)")
                print("Added new calculated HR channel")

        # Сохраняем данные
        np.save(
            str(path_to_save / f"{file_name}_processed.npy"),
            {
                "signals": processed_signals,
                "labels": ch_new,
                "sampling_rate": samp_freq,
            },
        )

        # === ВИЗУАЛИЗАЦИЯ КАНАЛОВ (С ИНТЕРВАЛОМ) ===
        print("Creating channels visualization...")
        n_channels = len(processed_signals)

        # Проверяем, что есть каналы
        if n_channels == 0:
            print("No channels to plot!")
            continue

        # Рассчитываем индексы для временного интервала
        start_index = int(start_time_sec * samp_freq)
        end_index = int(end_time_sec * samp_freq)

        # Проверяем корректность интервала
        signal_length = len(processed_signals[0])
        if start_index < 0:
            start_index = 0
        if end_index > signal_length:
            end_index = signal_length
        if start_index >= end_index:
            print(
                f"Warning: invalid time interval [{start_time_sec}, {end_time_sec}]. Using default [0, 30] sec."
            )
            start_index = 0
            end_index = min(30 * samp_freq, signal_length)

        # Создаем график с несколькими subplots
        fig, axes = plt.subplots(
            n_channels, 1, figsize=(15, 2 * n_channels), sharex=True
        )
        fig.suptitle(
            f"Channels visualization: {file_name} ({start_time_sec}-{end_time_sec} sec)",
            fontsize=16,
        )

        # Если только 1 канал - делаем axes массивом для единообразия
        if n_channels == 1:
            axes = [axes]

        # Временная ось для выбранного интервала
        time_axis = np.arange(start_index, end_index) / samp_freq

        for i in range(n_channels):
            # Проверяем длину текущего канала
            ch_length = len(processed_signals[i])

            # Корректируем индексы для текущего канала
            ch_start = min(start_index, ch_length)
            ch_end = min(end_index, ch_length)

            # Берем только указанный интервал
            display_signal = processed_signals[i][ch_start:ch_end]

            # Корректируем временную ось для текущего канала
            ch_time_axis = time_axis[: len(display_signal)]

            axes[i].plot(ch_time_axis, display_signal)
            axes[i].set_ylabel(ch_new[i], rotation=0, labelpad=40, ha="right")
            axes[i].grid(True)

        plt.xlabel("Time (seconds)")
        plt.xlim(start_time_sec, end_time_sec)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Учитываем заголовок

        # Сохраняем в файл
        plot_filename = (
            path_to_save
            / f"{file_name}_channels_plot_{start_time_sec}_{end_time_sec}sec.png"
        )
        plt.savefig(str(plot_filename), dpi=100)
        plt.close(fig)
        print(f"Saved channel plot: {plot_filename}")

print("\nAll files processed successfully!")
