"""Wspólny moduł wskaźników analizy technicznej."""

import pandas as pd


def oblicz_sma(data, window):
    """Oblicza prostą średnią kroczącą (SMA) dla kolumny 'Close'."""
    return data['Close'].rolling(window=window).mean()


def oblicz_rsi(data, period=14):
    """Oblicza wskaźnik RSI (Relative Strength Index)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def oblicz_macd(data, span_short=12, span_long=26, span_signal=9):
    """Oblicza MACD i linię sygnałową."""
    exp_short = data['Close'].ewm(span=span_short, adjust=False).mean()
    exp_long = data['Close'].ewm(span=span_long, adjust=False).mean()
    macd = exp_short - exp_long
    signal_line = macd.ewm(span=span_signal, adjust=False).mean()
    return macd, signal_line


def oblicz_bollinger(data, window=20, num_std=2):
    """Oblicza wstęgi Bollingera."""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower


def dodaj_wszystkie_wskazniki(data):
    """Dodaje wszystkie wskaźniki techniczne do ramki danych."""
    data['SMA50'] = oblicz_sma(data, 50)
    data['SMA200'] = oblicz_sma(data, 200)
    data['RSI'] = oblicz_rsi(data)
    data['MACD'], data['Signal_Line'] = oblicz_macd(data)
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['SMA_diff'] = data['SMA50'] - data['SMA200']
    data['MACD_diff'] = data['MACD'] - data['Signal_Line']
    data['BB_Mid'], data['BB_Upper'], data['BB_Lower'] = oblicz_bollinger(data)

    # Cechy relatywne
    bb_range = data['BB_Upper'] - data['BB_Lower']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / bb_range.replace(0, float('nan'))
    data['SMA50_dist'] = (data['Close'] - data['SMA50']) / data['SMA50']
    data['SMA200_dist'] = (data['Close'] - data['SMA200']) / data['SMA200']
    data['ROC_10'] = data['Close'].pct_change(periods=10)
    data['ROC_30'] = data['Close'].pct_change(periods=30)

    return data
