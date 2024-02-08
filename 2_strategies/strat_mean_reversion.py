from dotenv import load_dotenv
import os
import ccxt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '1_indicators'))
import indicators as n

# Load env
load_dotenv()
bybt_key = os.getenv('BYBT_KEY')
bybt_secret = os.getenv('BYBT_SECRET')



bybit = ccxt.bybit({
  'enableRateLimit': True,
  'apiKey': bybt_key,
  'secret': bybt_secret
})

print(bybit.fetch_balance())