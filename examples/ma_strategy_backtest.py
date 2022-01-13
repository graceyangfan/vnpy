from zenquant.ctastrategy.backtesting import BacktestingEngine, OptimizationSetting
from zenquant.ctastrategy.strategies.ma_martingale import (
   MaMartingaleStrategy
)
from datetime import datetime

##注意start和end要在数据库中，
##调用utils中的download_binance 下载数据存储于/home/username/.vntrader
engine = BacktestingEngine()
engine.set_parameters(
    vt_symbol="ETHUSDT.BINANCE",
    interval="1m",
    start=datetime(2021, 11, 1),
    end=datetime(2021, 12, 1),
    rate=0.0004,
    slippage=0,
    size=1,
    pricetick=0.01,
    capital=100,
)
setting={
    "min_volume":0.001,
}
engine.add_strategy(MaMartingaleStrategy,setting )

engine.load_data()
engine.run_backtesting()
df = engine.calculate_result()
engine.calculate_statistics()
engine.show_chart()

setting = OptimizationSetting()
setting.set_target("sharpe_ratio")
setting.add_parameter("min_volume", 0.001,None,None)
setting.add_parameter("ma5_fast_window", 5, 15,1)
setting.add_parameter("ma5_slow_window", 15, 30,1)

engine.run_ga_optimization(setting)
