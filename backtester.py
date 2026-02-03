import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from indicators import dodaj_wszystkie_wskazniki

# --- Logika Przygotowania Danych i Wskaźników ---

def prepare_data_for_ml(ticker, okres='10y'):
    """
    Pobiera i przygotowuje dane, zwracając kompletną ramkę danych z cechami i celem.
    """
    print(f"Pobieranie danych i przygotowywanie dla {ticker}...")
    dane = yf.download(ticker, period=okres, auto_adjust=True)
    if dane.empty:
        return None

    # Oblicz wskaźniki za pomocą wspólnego modułu
    dane = dodaj_wszystkie_wskazniki(dane)

    # Definicja "Targetu"
    future_window = 30
    target_percentage = 1.05
    dane['Target'] = (dane['Close'].shift(-future_window) > dane['Close'] * target_percentage)

    # Ostateczne czyszczenie danych
    dane_finalne = dane.dropna()
    dane_finalne['Target'] = dane_finalne['Target'].astype(int)

    return dane_finalne

# --- Logika Backtestera ---

def _jako_float(wartosc):
    """Konwertuje wartość pandas/numpy na zwykły float Pythona."""
    if hasattr(wartosc, 'item'):
        return wartosc.item()
    elif hasattr(wartosc, 'values'):
        return float(wartosc.values.flatten()[0])
    return float(wartosc)

def uruchom_backtest_ml(dane_testowe, predykcje, kapital_poczatkowy=10000):
    """
    Przeprowadza symulację handlu na podstawie predykcji z modelu ML.
    """
    print("Uruchamianie symulacji (backtestu) na podstawie decyzji modelu ML...")

    gotowka = float(kapital_poczatkowy)
    ilosc_akcji = 0.0
    pozycja_otwarta = False

    for i in range(len(dane_testowe)):
        sygnal_kupna = predykcje[i] == 1
        sygnal_sprzedazy = predykcje[i] == 0
        aktualna_cena = _jako_float(dane_testowe['Close'].iloc[i])

        if sygnal_kupna and not pozycja_otwarta:
            ilosc_akcji = gotowka / aktualna_cena
            gotowka = 0.0
            pozycja_otwarta = True

        elif sygnal_sprzedazy and pozycja_otwarta:
            gotowka = ilosc_akcji * aktualna_cena
            ilosc_akcji = 0.0
            pozycja_otwarta = False

    ostatnia_cena = _jako_float(dane_testowe['Close'].iloc[-1])
    wartosc_koncowa = gotowka + ilosc_akcji * ostatnia_cena
    return float(wartosc_koncowa)

def main():
    TICKER = 'CDR.WA'
    OKRES = '10y'
    KAPITAL_POCZATKOWY = 10000

    # 1. Przygotowanie danych
    dane = prepare_data_for_ml(TICKER)
    if dane is None or dane.empty:
        print("Nie udało się przygotować danych.")
        return

    features = ['RSI', 'SMA_diff', 'MACD_diff', 'Volatility']
    X = dane[features]
    y = dane['Target']

    # 2. Podział danych na zbiór treningowy i testowy - WAŻNE: bez tasowania!
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    dane_testowe = dane[split_index:]

    print(f"\nPrzygotowano {len(X)} dni do analizy.")
    print(f"Zbiór treningowy: {len(X_train)} dni (do {X_train.index[-1].date()})")
    print(f"Zbiór testowy: {len(X_test)} dni (od {X_test.index[0].date()})")

    # 3. Trening modelu
    print("\nTrening modelu RandomForest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 4. Predykcja dla okresu testowego
    predykcje = model.predict(X_test)

    # 5. Uruchomienie symulacji na podstawie predykcji modelu
    koniec_ml = uruchom_backtest_ml(dane_testowe, predykcje, KAPITAL_POCZATKOWY)
    zwrot_ml = float((koniec_ml / KAPITAL_POCZATKOWY - 1) * 100)

    # 6. Obliczenie wyniku dla "Kup i Trzymaj" dla tego samego okresu testowego
    poczatkowa_cena = _jako_float(dane_testowe['Close'].iloc[0])
    koncowa_cena = _jako_float(dane_testowe['Close'].iloc[-1])
    kup_i_trzymaj_wartosc = (KAPITAL_POCZATKOWY / poczatkowa_cena) * koncowa_cena
    zwrot_bh = float((kup_i_trzymaj_wartosc / KAPITAL_POCZATKOWY - 1) * 100)

    # 7. Wyświetlanie wyników
    print("\n--- Zakończono Symulację ---")
    print(f"\nAnalizowany okres testowy: od {dane_testowe.index[0].date()} do {dane_testowe.index[-1].date()}")
    print(f"Kapitał początkowy: {KAPITAL_POCZATKOWY:.2f} PLN")

    print("\n--- Wyniki ---")
    print(f"Stopa zwrotu strategii ML: {zwrot_ml:.2f}%")
    print(f"Stopa zwrotu 'Kup i Trzymaj': {zwrot_bh:.2f}%")

    if zwrot_ml > zwrot_bh:
        print("\nGratulacje! Twoja strategia oparta na modelu ML okazała się lepsza niż pasywne 'Kup i Trzymaj'.")
    else:
        print("\nNiestety, w tym okresie strategia 'Kup i Trzymaj' okazała się bardziej opłacalna.")

if __name__ == "__main__":
    main()
