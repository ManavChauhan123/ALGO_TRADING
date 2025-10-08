# main.py - Enhanced FastAPI Backend for Algo Trading System
from dateutil import parser
import logging
import time
import os
from typing import Optional, Dict, List
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import yfinance as yf
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Alpaca API imports
try:
    from alpaca.data.live import StockDataStream
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.client import TradingClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("alpaca-py not installed. Install with: pip install alpaca-py")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Setup
engine = create_engine('sqlite:///trades.db', echo=False)
Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    type = Column(String)  # BUY/SELL
    price = Column(Float)
    shares = Column(Integer)
    value = Column(Float)
    pnl = Column(Float, nullable=True)
    timestamp = Column(DateTime)
    reason = Column(String)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# FastAPI App
app = FastAPI(title="Algo Trading Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class StrategyParams(BaseModel):
    sma_short: int = 20
    sma_long: int = 50
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    bb_period: int = 20
    bb_std: float = 2.0

class RiskManagement(BaseModel):
    max_position_size: float = 0.95
    stop_loss: float = 5.0
    take_profit: float = 10.0

class BacktestRequest(BaseModel):
    symbol: str
    strategy: str
    params: StrategyParams
    risk: RiskManagement
    start_date: str
    end_date: str

class TradeResponse(BaseModel):
    id: int
    symbol: str
    type: str
    price: float
    shares: int
    value: float
    pnl: Optional[float]
    timestamp: datetime
    reason: str

class TradeHistoryResponse(BaseModel):
    trades: List[TradeResponse]
    total_pnl: float
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float

# Helper Functions
def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(series: pd.Series, period: int, std: float) -> Dict[str, pd.Series]:
    sma = calculate_sma(series, period)
    rolling_std = series.rolling(window=period).std()
    return {
        'upper': sma + (rolling_std * std),
        'middle': sma,
        'lower': sma - (rolling_std * std)
    }

def calculate_performance_metrics(equity_curve: List[Dict], start_date: str, end_date: str) -> Dict:
    """Calculate Sharpe Ratio, Max Drawdown, and Annualized Returns"""
    if len(equity_curve) < 2:
        return {'sharpe_ratio': 0, 'max_drawdown': 0, 'annualized_return': 0}
    
    # Convert to DataFrame for easier calculations
    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate daily returns
    df['returns'] = df['value'].pct_change()
    daily_returns = df['returns'].dropna()
    
    # Sharpe Ratio (assuming risk-free rate = 0)
    if len(daily_returns) > 0 and daily_returns.std() != 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0
    
    # Maximum Drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100  # Convert to percentage
    
    # Annualized Returns
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    days = (end_dt - start_dt).days
    years = days / 365.25
    
    if years > 0:
        initial_value = df['value'].iloc[0]
        final_value = df['value'].iloc[-1]
        annualized_return = ((final_value / initial_value) ** (1 / years) - 1) * 100
    else:
        annualized_return = 0
    
    return {
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown': round(abs(max_drawdown), 2),
        'annualized_return': round(annualized_return, 2)
    }

def generate_signals(df: pd.DataFrame, strategy: str, params: StrategyParams) -> pd.DataFrame:
    df = df.copy()
    close = df['Close']
    
    if strategy == 'sma_crossover':
        df['short_sma'] = calculate_sma(close, params.sma_short)
        df['long_sma'] = calculate_sma(close, params.sma_long)
        df['signal'] = np.where(df['short_sma'] > df['long_sma'], 1, 0)
        df['signal'] = df['signal'].diff()
        df['buy'] = np.where(df['signal'] == 1, df['Close'], np.nan)
        df['sell'] = np.where(df['signal'] == -1, df['Close'], np.nan)
    
    elif strategy == 'rsi':
        df['rsi'] = calculate_rsi(close, params.rsi_period)
        df['buy'] = np.where(df['rsi'] < params.rsi_oversold, df['Close'], np.nan)
        df['sell'] = np.where(df['rsi'] > params.rsi_overbought, df['Close'], np.nan)
    
    elif strategy == 'bollinger':
        bb = calculate_bollinger_bands(close, params.bb_period, params.bb_std)
        df['bb_upper'] = bb['upper']
        df['bb_lower'] = bb['lower']
        df['buy'] = np.where(close < df['bb_lower'], df['Close'], np.nan)
        df['sell'] = np.where(close > df['bb_upper'], df['Close'], np.nan)
    
    return df

def run_backtest(symbol: str, strategy: str, params: StrategyParams, risk: RiskManagement, start_date: str, end_date: str) -> Dict:
    # Add .NS for Indian stocks if needed
    original_symbol = symbol
    if not any(x in symbol.upper() for x in ['USD', '.NS']) and symbol.upper() not in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']:
        symbol = symbol + '.NS'
    
    logger.info(f"Backtesting {symbol} from {start_date} to {end_date}")
    
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise ValueError(f"No data fetched for {symbol}")
    
    # Handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    df = generate_signals(df, strategy, params)
    
    cash = 100000.0
    position = 0
    entry_price = 0.0
    trades = []
    equity_curve = []
    buy_signals = []
    sell_signals = []
    
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i]
        
        # Risk Management Check
        if position > 0:
            pnl_pct = ((price - entry_price) / entry_price) * 100
            if pnl_pct <= -risk.stop_loss:
                revenue = position * price
                pnl = revenue - (position * entry_price)
                cash += revenue
                trades.append({
                    'type': 'SELL', 
                    'price': price, 
                    'shares': position, 
                    'pnl': pnl, 
                    'date': date, 
                    'reason': 'Stop Loss',
                    'confidence': 1.0
                })
                sell_signals.append({'date': date.strftime('%Y-%m-%d'), 'price': price, 'reason': 'Stop Loss'})
                position = 0
            elif pnl_pct >= risk.take_profit:
                revenue = position * price
                pnl = revenue - (position * entry_price)
                cash += revenue
                trades.append({
                    'type': 'SELL', 
                    'price': price, 
                    'shares': position, 
                    'pnl': pnl, 
                    'date': date, 
                    'reason': 'Take Profit',
                    'confidence': 1.0
                })
                sell_signals.append({'date': date.strftime('%Y-%m-%d'), 'price': price, 'reason': 'Take Profit'})
                position = 0
        
        # Strategy Signals
        if not pd.isna(df['buy'].iloc[i]) and position == 0:
            shares = int((cash * risk.max_position_size) / price)
            if shares > 0:
                cost = shares * price
                cash -= cost
                position = shares
                entry_price = price
                trades.append({
                    'type': 'BUY', 
                    'price': price, 
                    'shares': shares, 
                    'date': date, 
                    'reason': 'Strategy Buy',
                    'confidence': 0.8
                })
                buy_signals.append({'date': date.strftime('%Y-%m-%d'), 'price': price, 'reason': 'Strategy Buy'})
        
        elif not pd.isna(df['sell'].iloc[i]) and position > 0:
            revenue = position * price
            pnl = revenue - (position * entry_price)
            cash += revenue
            trades.append({
                'type': 'SELL', 
                'price': price, 
                'shares': position, 
                'pnl': pnl, 
                'date': date, 
                'reason': 'Strategy Sell',
                'confidence': 0.8
            })
            sell_signals.append({'date': date.strftime('%Y-%m-%d'), 'price': price, 'reason': 'Strategy Sell'})
            position = 0
        
        total_value = cash + (position * price)
        equity_curve.append({'date': date.strftime('%Y-%m-%d'), 'value': total_value})
    
    # Close any open position
    if position > 0:
        price = df['Close'].iloc[-1]
        revenue = position * price
        pnl = revenue - (position * entry_price)
        cash += revenue
        trades.append({
            'type': 'SELL', 
            'price': price, 
            'shares': position, 
            'pnl': pnl, 
            'date': df.index[-1], 
            'reason': 'End of Test',
            'confidence': 1.0
        })
        sell_signals.append({'date': df.index[-1].strftime('%Y-%m-%d'), 'price': price, 'reason': 'End of Test'})
    
    # Calculate metrics
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    total_trades = len(sell_trades)
    wins = [t for t in sell_trades if t['pnl'] > 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in sell_trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in sell_trades) else 0
    
    # Calculate performance metrics
    perf_metrics = calculate_performance_metrics(equity_curve, start_date, end_date)
    
    # Store trades in DB
    session = Session()
    for trade in trades:
        db_trade = Trade(
            symbol=original_symbol,
            type=trade['type'],
            price=trade['price'],
            shares=trade['shares'],
            value=trade['shares'] * trade['price'],
            pnl=trade.get('pnl'),
            timestamp=trade['date'],
            reason=trade['reason']
        )
        session.add(db_trade)
    session.commit()
    session.close()
    
    # Prepare data for frontend
    df_reset = df.reset_index()
    df_reset['Date'] = df_reset['Date'].astype(str)
    # Replace NaN and inf with None for JSON compatibility
    df_reset = df_reset.replace([np.nan, np.inf, -np.inf], None)
    
    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'metrics': {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_value': cash,
            'sharpe_ratio': perf_metrics['sharpe_ratio'],
            'max_drawdown': perf_metrics['max_drawdown'],
            'annualized_return': perf_metrics['annualized_return']
        },
        'data': df_reset.to_dict(orient='records')
    }

# API Endpoints

@app.get("/historical/{symbol}")
async def get_historical(symbol: str, start_date: str, end_date: Optional[str] = None):
    try:
        original_symbol = symbol
        if not any(x in symbol.upper() for x in ['USD', '.NS']) and symbol.upper() not in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']:
            symbol = symbol + '.NS'
        
        logger.info(f"Fetching data for {symbol} (original: {original_symbol})")
        
        start_dt = parser.parse(start_date).strftime('%Y-%m-%d')
        end_dt = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d') if end_date is None else parser.parse(end_date).strftime('%Y-%m-%d')
        
        df = None
        for attempt in range(3):
            try:
                df = yf.download(symbol, start=start_dt, end=end_dt, progress=False)
                if not df.empty:
                    break
                time.sleep(2 ** attempt)
            except Exception as retry_err:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise HTTPException(status_code=503, detail=f"yfinance failed: {str(retry_err)}")
        
        if df is None or df.empty:
            raise HTTPException(status_code=503, detail=f"No data for {symbol}")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        df = df.reset_index()
        df['Date'] = df['Date'].astype(str)
        
        return df.to_dict(orient='records')
    
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest")
async def backtest(req: BacktestRequest):
    try:
        result = run_backtest(req.symbol, req.strategy, req.params, req.risk, req.start_date, req.end_date)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/trades/{symbol}")
async def get_trades(symbol: str):
    """Get all trades for a specific symbol with aggregated metrics"""
    session = Session()
    
    # Fetch all trades for the symbol
    trades = session.query(Trade).filter(Trade.symbol == symbol).order_by(Trade.timestamp.desc()).all()
    
    # Calculate aggregated metrics
    sell_trades = [t for t in trades if t.type == 'SELL' and t.pnl is not None]
    total_trades = len(sell_trades)
    total_pnl = sum(t.pnl for t in sell_trades)
    
    wins = [t for t in sell_trades if t.pnl > 0]
    losses = [t for t in sell_trades if t.pnl < 0]
    
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    
    session.close()
    
    trade_responses = [
        TradeResponse(
            id=t.id, 
            symbol=t.symbol, 
            type=t.type, 
            price=t.price, 
            shares=t.shares, 
            value=t.value, 
            pnl=t.pnl, 
            timestamp=t.timestamp, 
            reason=t.reason
        ) for t in trades
    ]
    
    return TradeHistoryResponse(
        trades=trade_responses,
        total_pnl=total_pnl,
        total_trades=total_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss
    )

@app.get("/predict/{symbol}")
async def predict_price(symbol: str):
    try:
        end = datetime.now()
        start = end - timedelta(days=365)
        df = yf.download(symbol, start=start, end=end, progress=False)
        
        if df.empty:
            raise ValueError("No data for ML training")
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        
        last_60 = scaled_data[-60:]
        pred_input = np.reshape(last_60, (1, 60, 1))
        predicted = model.predict(pred_input, verbose=0)
        predicted_price = scaler.inverse_transform(predicted)[0][0]
        
        return {'predicted_price': float(predicted_price)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# WebSocket for Live Data with Alpaca API
@app.websocket("/ws/live/{symbol}")
async def websocket_live(websocket: WebSocket, symbol: str):
    await websocket.accept()
    
    # Check if Alpaca is configured
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    if ALPACA_AVAILABLE and api_key and api_secret:
        logger.info(f"Using Alpaca API for real-time data: {symbol}")
        try:
            # Initialize Alpaca clients
            trading_client = TradingClient(api_key, api_secret, paper=True)
            data_client = StockHistoricalDataClient(api_key, api_secret)
            
            while True:
                try:
                    # Fetch latest quote
                    request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                    latest_quote = data_client.get_stock_latest_quote(request_params)
                    
                    quote = latest_quote[symbol]
                    
                    # Get current positions
                    try:
                        positions = trading_client.get_all_positions()
                        position = next((p for p in positions if p.symbol == symbol), None)
                        
                        position_data = {
                            'shares': float(position.qty) if position else 0,
                            'entry_price': float(position.avg_entry_price) if position else 0,
                            'current_price': float(quote.ask_price),
                            'market_value': float(position.market_value) if position else 0
                        }
                    except:
                        position_data = {
                            'shares': 0,
                            'entry_price': 0,
                            'current_price': float(quote.ask_price),
                            'market_value': 0
                        }
                    
                    # Get account info
                    try:
                        account = trading_client.get_account()
                        account_data = {
                            'cash': float(account.cash),
                            'portfolio_value': float(account.portfolio_value),
                            'buying_power': float(account.buying_power)
                        }
                    except:
                        account_data = {'cash': 0, 'portfolio_value': 0, 'buying_power': 0}
                    
                    data = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'price': float(quote.ask_price),
                        'bid': float(quote.bid_price),
                        'ask': float(quote.ask_price),
                        'position': position_data,
                        'account': account_data,
                        'source': 'alpaca'
                    }
                    
                    await websocket.send_json(data)
                    await asyncio.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Alpaca streaming error: {str(e)}")
                    await asyncio.sleep(10)
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for {symbol}")
    else:
        # Fallback to yfinance simulation
        logger.info(f"Using yfinance simulation for {symbol} (Alpaca not configured)")
        try:
            while True:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    data = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                        'volume': info.get('volume', 0),
                        'source': 'yfinance_simulated'
                    }
                    
                    await websocket.send_json(data)
                    await asyncio.sleep(60)  # Update every minute
                    
                except Exception as e:
                    logger.error(f"yfinance error: {str(e)}")
                    await asyncio.sleep(60)
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for {symbol}")

# Health check
@app.get("/")
async def root():
    return {
        "status": "online",
        "alpaca_available": ALPACA_AVAILABLE,
        "alpaca_configured": bool(os.getenv('ALPACA_API_KEY'))
    }

# Run with: uvicorn main:app --reload   ``