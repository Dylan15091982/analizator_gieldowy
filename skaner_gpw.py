import yfinance as yf
import time
from indicators import oblicz_sma, oblicz_rsi

def analizuj_spolki(tickers):
    """
    Skanuje podane tickery w poszukiwaniu sygnałów transakcyjnych:
    - Złoty krzyż / Krzyż śmierci
    - Poziomy RSI (wykupienie/wyprzedanie)
    """
    print("Skanowanie rynku w poszukiwaniu sygnałów transakcyjnych...")
    sygnaly_kupna = []
    sygnaly_sprzedazy = []
    spolki_wyprzedane = []
    spolki_wykupione = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, period='1y', progress=False, auto_adjust=True)
            
            if len(data) < 200:
                continue

            # Obliczenia wskaźników
            data['SMA50'] = oblicz_sma(data, 50)
            data['SMA200'] = oblicz_sma(data, 200)
            data['RSI'] = oblicz_rsi(data)
            
            # --- Sprawdzanie sygnałów ---
            
            # 1. Sygnały przecięcia średnich
            ostatnie_dni = data.tail(5)
            if (ostatnie_dni['SMA50'] > ostatnie_dni['SMA200']).any() and \
               (ostatnie_dni['SMA50'].iloc[-5] < ostatnie_dni['SMA200'].iloc[-5]):
                print(f"  -> Sygnał KUPNA (Złoty Krzyż) dla: {ticker}")
                sygnaly_kupna.append(ticker)
            elif (ostatnie_dni['SMA50'] < ostatnie_dni['SMA200']).any() and \
                 (ostatnie_dni['SMA50'].iloc[-5] > ostatnie_dni['SMA200'].iloc[-5]):
                print(f"  -> Sygnał SPRZEDAŻY (Krzyż Śmierci) dla: {ticker}")
                sygnaly_sprzedazy.append(ticker)
                
            # 2. Sygnały RSI
            ostatni_rsi = data['RSI'].iloc[-1]
            if ostatni_rsi < 30:
                print(f"  -> Spółka WYPRZEDANA (RSI={ostatni_rsi:.2f}) dla: {ticker}")
                spolki_wyprzedane.append(f"{ticker} (RSI: {ostatni_rsi:.2f})")
            elif ostatni_rsi > 70:
                print(f"  -> Spółka WYKUPIONA (RSI={ostatni_rsi:.2f}) dla: {ticker}")
                spolki_wykupione.append(f"{ticker} (RSI: {ostatni_rsi:.2f})")

            time.sleep(0.1)

        except Exception as e:
            pass
            
    return sygnaly_kupna, sygnaly_sprzedazy, spolki_wyprzedane, spolki_wykupione

def main():
    """Główna funkcja skanera."""
    
    lista_tickerow_wig20 = [
        'ALE.WA', 'ALR.WA', 'BDX.WA', 'CCC.WA', 'CDR.WA', 'DNP.WA', 'KGH.WA', 
        'KRU.WA', 'KTY.WA', 'LPP.WA', 'MBK.WA', 'OPL.WA', 'PCO.WA', 'PEO.WA', 
        'PGE.WA', 'PKN.WA', 'PKO.WA', 'PZU.WA', 'SPL.WA', 'ZAB.WA'
    ]
    
    print(f"Rozpoczynam analizę dla {len(lista_tickerow_wig20)} spółek z indeksu WIG20.")
    
    sygnaly_kupna, sygnaly_sprzedazy, spolki_wyprzedane, spolki_wykupione = analizuj_spolki(lista_tickerow_wig20)
    
    print("\n--- Podsumowanie ---")

    if spolki_wyprzedane:
        print("\nSpółki potencjalnie WYPRZEDANE (RSI < 30) - sygnał do obserwacji pod kątem kupna:")
        for ticker in spolki_wyprzedane:
            print(f"- {ticker}")
    else:
        print("\nNie znaleziono spółek w strefie wyprzedania (RSI < 30).")
        
    if spolki_wykupione:
        print("\nSpółki potencjalnie WYKUPIONE (RSI > 70) - sygnał do obserwacji pod kątem sprzedaży:")
        for ticker in spolki_wykupione:
            print(f"- {ticker}")
    else:
        print("\nNie znaleziono spółek w strefie wykupienia (RSI > 70).")

    if sygnaly_kupna:
        print("\nSpółki z potencjalnym sygnałem KUPNA (Złoty Krzyż):")
        for ticker in sygnaly_kupna:
            print(f"- {ticker}")
    else:
        print("\nNie znaleziono spółek z sygnałem 'Złotego Krzyża'.")
        
    if sygnaly_sprzedazy:
        print("\nSpółki z potencjalnym sygnałem SPRZEDAŻY (Krzyż Śmierci):")
        for ticker in sygnaly_sprzedazy:
            print(f"- {ticker}")
    else:
        print("\nNie znaleziono spółek z sygnałem 'Krzyża Śmierci'.")

if __name__ == "__main__":
    main()