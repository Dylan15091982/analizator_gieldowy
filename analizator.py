import logging
import os
import argparse

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from indicators import dodaj_wszystkie_wskazniki

logger = logging.getLogger(__name__)


def pobierz_dane(ticker, period):
    """Pobiera dane historyczne dla danego symbolu i okresu, z obsługą błędów."""
    logger.info("Pobieranie danych dla symbolu: %s za okres: %s...", ticker, period)
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            logger.warning("Nie znaleziono danych dla symbolu: %s.", ticker)
            return None
        # Spłaszcz MultiIndex kolumn (yfinance zwraca je dla pojedynczego tickera)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        logger.info("Dane zostały pomyślnie pobrane.")
        return data
    except Exception as e:
        logger.error("Wystąpił błąd podczas pobierania danych: %s", e)
        return None


def stworz_wykres(data, ticker, output_filename):
    """Generuje wielopanelowy wykres analizy technicznej z RSI, MACD, Bollinger Bands i wolumenem."""
    logger.info("Generowanie wykresu...")
    plt.style.use('seaborn-v0_8-darkgrid')

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    fig.suptitle(f'Analiza Techniczna Akcji: {ticker}', fontsize=18, y=0.95)

    # --- Panel 1: Cena + SMA + Bollinger Bands ---
    ax_price = axes[0]
    ax_price.plot(data.index, data['Close'], label='Cena zamknięcia', color='skyblue', linewidth=1.5)
    ax_price.plot(data.index, data['SMA50'], label='SMA 50', color='orange', linestyle='--', linewidth=1)
    ax_price.plot(data.index, data['SMA200'], label='SMA 200', color='red', linestyle='--', linewidth=1)
    ax_price.plot(data.index, data['BB_Upper'], color='gray', linestyle=':', linewidth=0.8)
    ax_price.plot(data.index, data['BB_Lower'], color='gray', linestyle=':', linewidth=0.8)
    ax_price.fill_between(data.index, data['BB_Upper'], data['BB_Lower'],
                          alpha=0.1, color='gray', label='Bollinger Bands')
    ax_price.set_ylabel('Cena')
    ax_price.legend(fontsize=8, loc='upper left')
    ax_price.grid(True)

    # --- Panel 2: Wolumen ---
    ax_vol = axes[1]
    colors = ['green' if c >= o else 'red'
              for c, o in zip(data['Close'], data['Open'])]
    ax_vol.bar(data.index, data['Volume'], color=colors, alpha=0.7, width=1)
    ax_vol.set_ylabel('Wolumen')
    ax_vol.grid(True)

    # --- Panel 3: RSI ---
    ax_rsi = axes[2]
    ax_rsi.plot(data.index, data['RSI'], color='purple', linewidth=1)
    ax_rsi.axhline(70, color='red', linestyle='--', linewidth=0.8)
    ax_rsi.axhline(30, color='green', linestyle='--', linewidth=0.8)
    ax_rsi.fill_between(data.index, 70, 100, alpha=0.1, color='red')
    ax_rsi.fill_between(data.index, 0, 30, alpha=0.1, color='green')
    ax_rsi.set_ylabel('RSI')
    ax_rsi.set_ylim(0, 100)
    ax_rsi.grid(True)

    # --- Panel 4: MACD ---
    ax_macd = axes[3]
    ax_macd.plot(data.index, data['MACD'], label='MACD', color='blue', linewidth=1)
    ax_macd.plot(data.index, data['Signal_Line'], label='Sygnał', color='orange', linewidth=1)
    histogram = data['MACD'] - data['Signal_Line']
    ax_macd.bar(data.index, histogram, color=['green' if v >= 0 else 'red' for v in histogram],
                alpha=0.5, width=1)
    ax_macd.set_ylabel('MACD')
    ax_macd.set_xlabel('Data')
    ax_macd.legend(fontsize=8, loc='upper left')
    ax_macd.grid(True)

    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)

    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Wykres zapisany jako: %s", output_path)


def main():
    """Główna funkcja skryptu."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description='Przeprowadza analizę techniczną akcji dla podanego symbolu.')
    parser.add_argument('ticker', type=str, help='Symbol giełdowy akcji (np. AAPL, GOOGL)')
    parser.add_argument('--period', type=str, default='5y',
                        help='Okres pobierania danych (np. 1y, 5y, 10y, max). Domyślnie: 5y')
    parser.add_argument('--output', type=str,
                        help='Nazwa pliku wyjściowego z wykresem (np. moja_analiza.png). Domyślnie: <TICKER>_stock_analysis.png')

    args = parser.parse_args()

    ticker = args.ticker
    period = args.period
    output_filename = args.output if args.output else f'{ticker}_stock_analysis.png'

    dane = pobierz_dane(ticker, period)
    if dane is not None:
        dane = dodaj_wszystkie_wskazniki(dane)
        stworz_wykres(dane, ticker, output_filename)


if __name__ == "__main__":
    main()
