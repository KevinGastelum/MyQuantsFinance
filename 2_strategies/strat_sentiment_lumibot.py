import os
from dotenv import load_dotenv
from datetime import datetime
from math import floor
from lumibot.strategies.strategy import Strategy
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from lumibot.entities import Asset
from lumibot.backtesting import YahooDataBacktesting
from alpaca_trade_api import REST
from timedelta import Timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple 
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Load APIs
load_dotenv()
API_KEY = os.getenv('ALPACA_KEY')
API_SECRET = os.getenv('ALPACA_SECRET')
BASE_URL = "https://paper-api.alpaca.markets"
HF_KEY = os.getenv('HF_KEY')

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

# Set Pytorch to use Cuda Tensors for GPU Processing
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# FinBert Model from HuggingFace for Sentiment Analysis on Stocks News 
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]


# Strategy
class MLSentiment(Strategy):
  def initialize(self, symbol:str="SPY", cash_at_risk:float=.5):
    self.symbol = symbol
    self.sleeptime = "24H" # Adjust SleepTime
    self.last_trade = None
    self.cash_at_risk = cash_at_risk # Currently set at 1%
    self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

# Returns Cash bal, Price of asset, Qty to buy == %1 of my bal
  def position_sizing(self):
    cash = self.get_cash() # My Balance
    last_price = self.get_last_price(self.symbol)
    quantity = floor(cash * self.cash_at_risk / last_price) # ADD leverage * calc
    return cash, last_price, quantity


  # Set Dates / Get News / Calc Sentiment
  def get_dates(self):
    today = self.get_datetime()
    x_days_prior = today - Timedelta(days=3) # News outlook days
    return today.strftime('%Y-%m-%d'), x_days_prior.strftime('%Y-%m-%d')

  def get_sentiment(self):
    today, x_days_prior = self.get_dates()
    news = self.api.get_news(symbol=self.symbol, start=x_days_prior, end=today)
    news = [ev.__dict__["_raw"]["headline"] for ev in news]
    probability, sentiment = estimate_sentiment(news)
    return probability, sentiment


# Initialize Trade order
  def on_trading_iteration(self):
    cash, last_price, quantity = self.position_sizing()
    probability, sentiment = self.get_sentiment()
    # base = Asset(symbol=self.symbol, asset_type='crypto')
    # quote = Asset("USD", asset_type='crypto')

    # Make sure we have enough funds to exec trade
    if cash > last_price:
        if sentiment == "positive" and probability > .999:
            if self.last_trade == "sell":
                  self.sell_all()
            order = self.create_order( # SELL if Sentiment Negative
                self.symbol,
                # base,
                quantity,
                "buy",
                type="bracket",
                # quote=quote,
                take_profit_price=last_price * 1.20, # TakeProfit 4 BUY = %20
                stop_loss_price=last_price * .95 # StopLoss = %5 increase
            )
            self.submit_order(order)
            self.last_trade = "buy"
        elif sentiment == "negative" and probability > .999: # BUY if Sentiment Positive
            if self.last_trade == "buy":
                  self.sell_all()
            order = self.create_order(
                self.symbol,
                # base,
                quantity,
                "sell",
                type="bracket",
                # quote=quote,
                take_profit_price=last_price * .8, # TakeProfit 4 SELL = %20
                stop_loss_price=last_price * 1.05 # StopLoss = %5 drop
            )
            self.submit_order(order)
            self.last_trade = "sell"


# Date range && Broker // alpaca == BTCUSD
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 2, 20)
broker = Alpaca(ALPACA_CREDS)
strategy = MLSentiment(name='mlsentiment', broker=broker,
                    parameters={"symbol":"SPY",
                                "cash_at_risk":.5})

# Backtest      // Note: BTC-USD Yfinance  
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol":"SPY", "cash_at_risk":.5}
    )
