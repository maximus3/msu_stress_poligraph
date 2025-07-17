#!/usr/bin/env python3
"""
Скрипт для быстрой настройки проекта MSU Stress Poligraph с UV
"""

import subprocess
import sys
import os
import platform


def run_command(cmd: str, check: bool = True) -> bool:
    """Выполнить команду и вывести результат"""
    print(f"Выполняю: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, check=check, capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and check:
            print(f"Предупреждение: {result.stderr}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении команды: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False


def check_uv_installed() -> bool:
    """Проверить, установлен ли UV"""
    return run_command("uv --version", check=False)


def install_uv() -> bool:
    """Установить UV в зависимости от операционной системы"""
    system = platform.system().lower()

    if system == "windows":
        print("Установка UV для Windows...")
        # Попробуем через pip как самый универсальный способ
        if not run_command(
            'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
            check=False,
        ):
            if not run_command("pip install uv", check=False):
                print("Не удалось установить. Попробуйте вручную:")
                print(
                    'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
                )
                print("Или через pip:")
                print("pip install uv")
                return False
    else:
        print("Установка UV для macOS/Linux...")
        if not run_command(
            "curl -LsSf https://astral.sh/uv/install.sh | sh", check=False
        ):
            print("Не удалось установить через curl. Попробуйте через pip:")
            print("pip install uv")
            return False

    return True


def setup_project() -> bool:
    """Настроить проект"""
    print("Настройка проекта MSU Stress Poligraph...")

    # Проверим, есть ли UV
    if not check_uv_installed():
        print("UV не найден. Устанавливаю...")
        if not install_uv():
            print("Не удалось установить UV. Установите вручную.")
            return False

    # Создаем виртуальное окружение и устанавливаем зависимости
    print("Создание виртуального окружения и установка зависимостей...")
    if not run_command("uv sync"):
        print("Не удалось создать окружение. Попробуйте вручную: uv sync")
        return False

    print("✅ Проект успешно настроен!")
    print("\nДля запуска используйте:")
    print("  uv run python main/lunohod_rest_work.py")
    print("\nИли активируйте окружение:")

    system = platform.system().lower()
    if system == "windows":
        print("  .venv\\Scripts\\activate")
    else:
        print("  source .venv/bin/activate")

    return True


def main() -> int:
    """Основная функция"""
    print("=== Настройка проекта MSU Stress Poligraph ===\n")

    # Проверим, что мы в правильной директории
    if not os.path.exists("pyproject.toml"):
        print("Ошибка: файл pyproject.toml не найден.")
        print("Убедитесь, что вы находитесь в корневой директории проекта.")
        return 1

    if setup_project():
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
