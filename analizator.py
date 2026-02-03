import yfinance as yf
import matplotlib.pyplot as plt
import os
import argparse
from indicators import oblicz_sma

def pobierz_dane(ticker, period):
    """Pobiera dane historyczne dla danego symbolu i okresu, z obsługą błędów."""
    print(f"Pobieranie danych dla symbolu: {ticker} za okres: {period}...")
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            print(f"Nie znaleziono danych dla symbolu: {ticker}. Może to być błędny symbol lub brak danych dla podanego okresu.")
            return None
        print("Dane zostały pomyślnie pobrane.")
        return data
    except Exception as e:
        print(f"Wystąpił błąd podczas pobierania danych: {e}")
        return None

def oblicz_wskazniki(data):
    """Oblicza wskaźniki analizy technicznej (SMA50, SMA200)."""
    print("Obliczanie wskaźników...")
    data['SMA50'] = oblicz_sma(data, 50)
    data['SMA200'] = oblicz_sma(data, 200)
    return data

def stworz_wykres(data, ticker, output_filename):
    """Generuje i zapisuje wykres analizy technicznej."""
    print("Generowanie wykresu...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(data['Close'], label='Cena zamknięcia', color='skyblue', linewidth=2)
    ax.plot(data['SMA50'], label='Średnia krocząca 50-dniowa', color='orange', linestyle='--')
    ax.plot(data['SMA200'], label='Średnia krocząca 200-dniowa', color='red', linestyle='--')

    ax.set_title(f'Analiza Techniczna Akcji: {ticker}', fontsize=18)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Cena (USD)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)
    
    plt.savefig(output_path)
    print(f"Wykres został pomyślnie zapisany jako: {output_path}")

def main():
    """Główna funkcja skryptu."""
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
        dane = oblicz_wskazniki(dane)
        stworz_wykres(dane, ticker, output_filename)

if __name__ == "__main__":
    main()