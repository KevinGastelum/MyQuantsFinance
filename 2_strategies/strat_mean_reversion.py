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






'''
MEAN REVERSION ---- 1

NICE FUNCS -- nice_funcs py 1.5hr May17, 22  1yr
https://www.youtube.com/watch?v=bD2T-R__Gj0

BRKT BOT -- m17 breakout.py 2.5hr May18, 22 1yr
https://www.youtube.com/watch?v=DaswM44ZvI8

RISK MNGMT -- m20_riskbot.py 5hr May23, 22 1yr
https://www.youtube.com/watch?v=Dvxt2tgDJ6s

78 CRYPTOS -- m23_phe_alltickers.py 3.4hr May24, 22
https://www.youtube.com/watch?v=etzNtgFaEyc

75 MeanREV -- m23_phe_alltick 3.1hr May25, 22 1yr
https://www.youtube.com/watch?v=tnjgRr5tiXc

Alg MeanREV -- m25_mn_rev_allphe 1hr May27, 22
https://www.youtube.com/watch?v=Rlu82lfU42w

AutomatedMnRv - ju29_meanreversion 1h Jul29, 22
https://www.youtube.com/watch?v=M9QZsyCSRzM

How Cde MenRevBot- ylive_Ju29_meanrev 30 Aug15,22
https://www.youtube.com/watch?v=Rq7Cy8N3b7w

HowCdeMenRevStra - Ylive_ju29_meanrev 12 Aug30, 22
https://www.youtube.com/watch?v=76Q7gLOFnbw

MeanRevTradBot - s17_mean_rev 15m Sep17, 22
https://www.youtube.com/watch?v=ziyLHTpEn24


nice_funcs
mean_reversion.py
ylive_ju29_meanreversion
ju_29handbot2
0 to Quant -- a1_recen_trade_total 8hr Jan20, 23
https://www.youtube.com/watch?v=A-iYBnGpM7I



BACKTESTING ---- 2

GPT BCKTST TRAD STRAT -Jan1, 23 - 1-9-bt py - 1yr ago
https://www.youtube.com/watch?v=73XcFcBic50


100 AI IS DOI JOB- ai15_onefile - Dec12, 23 - 5hrs - 1mo
https://www.youtube.com/watch?v=_gGdh6S6fvc




'''