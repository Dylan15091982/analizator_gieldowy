import logging
import csv
import os
from datetime import datetime

import yfinance as yf
import pandas as pd
import time
from indicators import oblicz_sma, oblicz_rsi

logger = logging.getLogger(__name__)


def analizuj_spolki(tickers):
    """
    Skanuje podane tickery w poszukiwaniu sygnałów transakcyjnych:
    - Złoty krzyż / Krzyż śmierci
    - Poziomy RSI (wykupienie/wyprzedanie)

    Zwraca krotkę: (sygnaly_kupna, sygnaly_sprzedazy, spolki_wyprzedane, spolki_wykupione, bledy)
    gdzie bledy to lista słowników {'ticker': ..., 'error': ...}.
    """
    logger.info("Skanowanie rynku w poszukiwaniu sygnałów transakcyjnych...")
    sygnaly_kupna = []
    sygnaly_sprzedazy = []
    spolki_wyprzedane = []
    spolki_wykupione = []
    bledy = []

    if not tickers:
        logger.warning("Lista tickerów jest pusta — nie ma nic do przeskanowania.")
        return sygnaly_kupna, sygnaly_sprzedazy, spolki_wyprzedane, spolki_wykupione, bledy

    for ticker in tickers:
        try:
            data = yf.download(ticker, period='1y', progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if data.empty:
                logger.warning("Brak danych dla %s, pomijam.", ticker)
                bledy.append({'ticker': ticker, 'error': 'Brak danych z yfinance'})
                continue

            if len(data) < 200:
                logger.debug("Za mało danych dla %s (%d dni), pomijam.", ticker, len(data))
                continue

            # Obliczenia wskaźników
            data['SMA50'] = oblicz_sma(data, 50)
            data['SMA200'] = oblicz_sma(data, 200)
            data['RSI'] = oblicz_rsi(data)

            # --- Sprawdzanie sygnałów ---

            # 1. Sygnały przecięcia średnich
            ostatnie_dni = data.tail(5)
            if len(ostatnie_dni) < 5:
                logger.debug("Za mało ostatnich dni dla %s, pomijam sygnały SMA.", ticker)
            else:
                if (ostatnie_dni['SMA50'] > ostatnie_dni['SMA200']).any() and \
                   (ostatnie_dni['SMA50'].iloc[-5] < ostatnie_dni['SMA200'].iloc[-5]):
                    logger.info("  -> Sygnał KUPNA (Złoty Krzyż) dla: %s", ticker)
                    sygnaly_kupna.append(ticker)
                elif (ostatnie_dni['SMA50'] < ostatnie_dni['SMA200']).any() and \
                     (ostatnie_dni['SMA50'].iloc[-5] > ostatnie_dni['SMA200'].iloc[-5]):
                    logger.info("  -> Sygnał SPRZEDAŻY (Krzyż Śmierci) dla: %s", ticker)
                    sygnaly_sprzedazy.append(ticker)

            # 2. Sygnały RSI
            ostatni_rsi = data['RSI'].iloc[-1]
            if pd.isna(ostatni_rsi):
                logger.debug("RSI jest NaN dla %s, pomijam.", ticker)
            elif ostatni_rsi < 30:
                logger.info("  -> Spółka WYPRZEDANA (RSI=%.2f) dla: %s", ostatni_rsi, ticker)
                spolki_wyprzedane.append({'ticker': ticker, 'rsi': float(ostatni_rsi)})
            elif ostatni_rsi > 70:
                logger.info("  -> Spółka WYKUPIONA (RSI=%.2f) dla: %s", ostatni_rsi, ticker)
                spolki_wykupione.append({'ticker': ticker, 'rsi': float(ostatni_rsi)})

            time.sleep(0.1)

        except Exception as e:
            logger.warning("Błąd podczas analizy %s: %s", ticker, e)
            bledy.append({'ticker': ticker, 'error': str(e)})

    return sygnaly_kupna, sygnaly_sprzedazy, spolki_wyprzedane, spolki_wykupione, bledy


def zapisz_wyniki_csv(sygnaly_kupna, sygnaly_sprzedazy, spolki_wyprzedane, spolki_wykupione, bledy=None):
    """Zapisuje wyniki skanowania (w tym błędy) do pliku CSV z datą."""
    data_skanowania = datetime.now().strftime('%Y-%m-%d_%H%M')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nazwa_pliku = os.path.join(script_dir, f'skan_{data_skanowania}.csv')

    try:
        with open(nazwa_pliku, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Data skanowania', 'Typ sygnału', 'Ticker', 'RSI'])

            for ticker in sygnaly_kupna:
                writer.writerow([data_skanowania, 'Złoty Krzyż (KUPNO)', ticker, ''])
            for ticker in sygnaly_sprzedazy:
                writer.writerow([data_skanowania, 'Krzyż Śmierci (SPRZEDAŻ)', ticker, ''])
            for s in spolki_wyprzedane:
                writer.writerow([data_skanowania, 'Wyprzedana (RSI<30)', s['ticker'], f"{s['rsi']:.2f}"])
            for s in spolki_wykupione:
                writer.writerow([data_skanowania, 'Wykupiona (RSI>70)', s['ticker'], f"{s['rsi']:.2f}"])
            for b in (bledy or []):
                writer.writerow([data_skanowania, f"BŁĄD: {b['error']}", b['ticker'], ''])

        logger.info("Wyniki zapisano do: %s", nazwa_pliku)
        return nazwa_pliku

    except OSError as e:
        logger.error("Nie udało się zapisać wyników do CSV: %s", e)
        return None


def main():
    """Główna funkcja skanera."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    lista_tickerow_wig20 = [
        'ALE.WA', 'ALR.WA', 'BDX.WA', 'CCC.WA', 'CDR.WA', 'DNP.WA', 'KGH.WA',
        'KRU.WA', 'KTY.WA', 'LPP.WA', 'MBK.WA', 'OPL.WA', 'PCO.WA', 'PEO.WA',
        'PGE.WA', 'PKN.WA', 'PKO.WA', 'PZU.WA', 'SPL.WA', 'ZAB.WA'
    ]

    logger.info("Rozpoczynam analizę dla %d spółek z indeksu WIG20.", len(lista_tickerow_wig20))

    try:
        sygnaly_kupna, sygnaly_sprzedazy, spolki_wyprzedane, spolki_wykupione, bledy = analizuj_spolki(
            lista_tickerow_wig20
        )
    except Exception as e:
        logger.critical("Nieoczekiwany błąd podczas skanowania: %s", e)
        return

    # Zapis do CSV
    zapisz_wyniki_csv(sygnaly_kupna, sygnaly_sprzedazy, spolki_wyprzedane, spolki_wykupione, bledy)

    # Podsumowanie na konsoli
    logger.info("\n--- Podsumowanie ---")

    if bledy:
        logger.warning("Błędy podczas skanowania (%d):", len(bledy))
        for b in bledy:
            logger.warning("- %s: %s", b['ticker'], b['error'])

    if spolki_wyprzedane:
        logger.info("Spółki potencjalnie WYPRZEDANE (RSI < 30):")
        for s in spolki_wyprzedane:
            logger.info("- %s (RSI: %.2f)", s['ticker'], s['rsi'])
    else:
        logger.info("Nie znaleziono spółek w strefie wyprzedania (RSI < 30).")

    if spolki_wykupione:
        logger.info("Spółki potencjalnie WYKUPIONE (RSI > 70):")
        for s in spolki_wykupione:
            logger.info("- %s (RSI: %.2f)", s['ticker'], s['rsi'])
    else:
        logger.info("Nie znaleziono spółek w strefie wykupienia (RSI > 70).")

    if sygnaly_kupna:
        logger.info("Spółki z sygnałem KUPNA (Złoty Krzyż):")
        for ticker in sygnaly_kupna:
            logger.info("- %s", ticker)
    else:
        logger.info("Nie znaleziono spółek z sygnałem 'Złotego Krzyża'.")

    if sygnaly_sprzedazy:
        logger.info("Spółki z sygnałem SPRZEDAŻY (Krzyż Śmierci):")
        for ticker in sygnaly_sprzedazy:
            logger.info("- %s", ticker)
    else:
        logger.info("Nie znaleziono spółek z sygnałem 'Krzyża Śmierci'.")


if __name__ == "__main__":
    main()
