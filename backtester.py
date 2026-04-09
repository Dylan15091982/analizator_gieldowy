import logging
import sys
import argparse

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix

from indicators import dodaj_wszystkie_wskazniki

logger = logging.getLogger(__name__)

PROWIZJA = 0.003  # 0.3% prowizji za transakcję

# --- Logika Przygotowania Danych i Wskaźników ---


def prepare_data_for_ml(ticker, okres='10y'):
    """
    Pobiera i przygotowuje dane, zwracając kompletną ramkę danych z cechami i celem.
    """
    if not ticker or not ticker.strip():
        logger.error("Symbol tickera nie może być pusty.")
        return None
    logger.info("Pobieranie danych i przygotowywanie dla %s...", ticker)
    try:
        dane = yf.download(ticker, period=okres, auto_adjust=True)
    except Exception as e:
        logger.error("Błąd podczas pobierania danych dla %s: %s", ticker, e)
        return None
    if dane.empty:
        logger.warning("Nie znaleziono danych dla symbolu: %s.", ticker)
        return None
    if isinstance(dane.columns, pd.MultiIndex):
        dane.columns = dane.columns.get_level_values(0)

    # Oblicz wskaźniki za pomocą wspólnego modułu
    try:
        dane = dodaj_wszystkie_wskazniki(dane)
    except (ValueError, TypeError) as e:
        logger.error("Błąd podczas obliczania wskaźników: %s", e)
        return None

    # Definicja "Targetu"
    future_window = 30
    target_percentage = 1.03
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


def uruchom_backtest_ml(dane_testowe, predykcje, kapital_poczatkowy=10000, prowizja=PROWIZJA):
    """
    Przeprowadza symulację handlu na podstawie predykcji z modelu ML.
    Uwzględnia koszty transakcyjne (prowizję).
    """
    logger.info("Uruchamianie backtestu (prowizja: %.2f%%)...", prowizja * 100)

    gotowka = float(kapital_poczatkowy)
    ilosc_akcji = 0.0
    pozycja_otwarta = False
    liczba_transakcji = 0

    for i in range(len(dane_testowe)):
        sygnal_kupna = predykcje[i] == 1
        sygnal_sprzedazy = predykcje[i] == 0
        aktualna_cena = _jako_float(dane_testowe['Close'].iloc[i])

        if sygnal_kupna and not pozycja_otwarta:
            koszt_prowizji = gotowka * prowizja
            gotowka -= koszt_prowizji
            ilosc_akcji = gotowka / aktualna_cena
            gotowka = 0.0
            pozycja_otwarta = True
            liczba_transakcji += 1

        elif sygnal_sprzedazy and pozycja_otwarta:
            gotowka = ilosc_akcji * aktualna_cena
            koszt_prowizji = gotowka * prowizja
            gotowka -= koszt_prowizji
            ilosc_akcji = 0.0
            pozycja_otwarta = False
            liczba_transakcji += 1

    ostatnia_cena = _jako_float(dane_testowe['Close'].iloc[-1])
    wartosc_koncowa = gotowka + ilosc_akcji * ostatnia_cena
    return float(wartosc_koncowa), liczba_transakcji


def walidacja_krzyzowa(X, y, features, n_splits=5):
    """Przeprowadza walidację krzyżową z TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    wyniki = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        wyniki.append(score)
        logger.info("  Fold %d: accuracy = %.4f", fold, score)

    srednia = np.mean(wyniki)
    odchylenie = np.std(wyniki)
    logger.info("Walidacja krzyżowa: %.4f (+/- %.4f)", srednia, odchylenie)
    return wyniki


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description='Backtester strategii ML na danych giełdowych.')
    parser.add_argument('ticker', type=str, nargs='?', default='CDR.WA',
                        help='Symbol giełdowy (np. CDR.WA, PKO.WA). Domyślnie: CDR.WA')
    parser.add_argument('--kapital', type=float, default=10000,
                        help='Kapitał początkowy w PLN. Domyślnie: 10000')
    args = parser.parse_args()

    TICKER = args.ticker
    KAPITAL_POCZATKOWY = args.kapital

    if KAPITAL_POCZATKOWY <= 0:
        logger.error("Kapitał początkowy musi być wartością dodatnią (podano: %.2f).", KAPITAL_POCZATKOWY)
        sys.exit(1)

    # 1. Przygotowanie danych
    dane = prepare_data_for_ml(TICKER)
    if dane is None or dane.empty:
        logger.error("Nie udało się przygotować danych dla %s. Przerywam.", TICKER)
        sys.exit(1)

    features = [
        'RSI', 'SMA_diff', 'MACD_diff', 'Volatility', 'BB_Position',
        'SMA50_dist', 'SMA200_dist', 'ROC_10', 'ROC_30', 'Returns',
    ]
    X = dane[features]
    y = dane['Target']

    # 2. Walidacja krzyżowa (TimeSeriesSplit)
    logger.info("\n--- Walidacja krzyżowa (5-fold TimeSeriesSplit) ---")
    walidacja_krzyzowa(X, y, features)

    # 3. Podział danych na zbiór treningowy i testowy
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    dane_testowe = dane[split_index:]

    logger.info("Przygotowano %d dni do analizy.", len(X))
    logger.info("Zbiór treningowy: %d dni (do %s)", len(X_train), X_train.index[-1].date())
    logger.info("Zbiór testowy: %d dni (od %s)", len(X_test), X_test.index[0].date())

    # 4. Trening modelu
    logger.info("Trening modelu RandomForest...")
    try:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
    except Exception as e:
        logger.error("Błąd podczas trenowania modelu: %s", e)
        sys.exit(1)

    # 5. Predykcja i metryki
    try:
        proba = model.predict_proba(X_test)[:, 1]
        predykcje = (proba >= 0.4).astype(int)
    except Exception as e:
        logger.error("Błąd podczas predykcji: %s", e)
        sys.exit(1)

    logger.info("\n--- Metryki klasyfikacji ---")
    logger.info("\n%s", classification_report(y_test, predykcje, target_names=['Spadek', 'Wzrost'], zero_division=0))
    logger.info("Macierz pomyłek:\n%s", confusion_matrix(y_test, predykcje))

    # 6. Ważność cech
    logger.info("\n--- Ważność cech ---")
    for nazwa, waznosc in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        logger.info("  %s: %.4f", nazwa, waznosc)

    # 7. Backtest z kosztami transakcyjnymi
    koniec_ml, transakcje = uruchom_backtest_ml(dane_testowe, predykcje, KAPITAL_POCZATKOWY)
    zwrot_ml = float((koniec_ml / KAPITAL_POCZATKOWY - 1) * 100)

    # 8. Obliczenie wyniku dla "Kup i Trzymaj"
    poczatkowa_cena = _jako_float(dane_testowe['Close'].iloc[0])
    koncowa_cena = _jako_float(dane_testowe['Close'].iloc[-1])
    kup_i_trzymaj_wartosc = (KAPITAL_POCZATKOWY / poczatkowa_cena) * koncowa_cena
    zwrot_bh = float((kup_i_trzymaj_wartosc / KAPITAL_POCZATKOWY - 1) * 100)

    # 9. Wyświetlanie wyników
    logger.info("\n--- Wyniki symulacji ---")
    logger.info("Okres testowy: od %s do %s", dane_testowe.index[0].date(), dane_testowe.index[-1].date())
    logger.info("Kapitał początkowy: %.2f PLN", KAPITAL_POCZATKOWY)
    logger.info("Liczba transakcji: %d (prowizja: %.1f%%)", transakcje, PROWIZJA * 100)
    logger.info("Stopa zwrotu strategii ML: %.2f%% (końcowa wartość: %.2f PLN)", zwrot_ml, koniec_ml)
    logger.info("Stopa zwrotu 'Kup i Trzymaj': %.2f%% (końcowa wartość: %.2f PLN)", zwrot_bh, kup_i_trzymaj_wartosc)

    if zwrot_ml > zwrot_bh:
        logger.info("Strategia ML okazała się lepsza niż pasywne 'Kup i Trzymaj'.")
    else:
        logger.info("Strategia 'Kup i Trzymaj' okazała się bardziej opłacalna.")


if __name__ == "__main__":
    main()
