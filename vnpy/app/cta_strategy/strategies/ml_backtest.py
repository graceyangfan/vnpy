import numpy as np 
from zenquant.ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
)
from zenquant.trader.constant import (
    Status,
    Direction,
    Offset,
    Exchange
)
import lightgbm as lgb
from zenquant.trader.utility import  round_to

from zenquant.feed.data import BarDataFeed 
from zenquant.env.observer import Observer 
from zenquant.utils.get_indicators_info import get_bar_level_indicator_info 

class MLStrategy(CtaTemplate):
    limit_total_margin_rate = 0.5  ##保证金和未成交订单的最大占用比率
    available_change_percent= 0.01  ##atr use percent 
    sma_window = 10 
    profit_stop_rate = 1.0
    loss_stop_rate = -0.3
    indicator_windows_list= [5,10,20,40]
    parameters = [ 
        "limit_total_margin_rate",
        "available_change_percent",
        "sma_window",
        "profit_stop_rate",
        "loss_stop_rate",
        "indicator_windows_list"]
    variables = []
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.vt_symbol = vt_symbol 
        ## params  for portfolio 
        self.MarginLevel = setting["MarginLevel"] 
        self.start_margin_rate = 1.0/self.MarginLevel 
        self.price_tick = 0.01
        self.min_volume = 0.001

        self.bg = BarGenerator(self.on_bar) 
        self.am = BarDataFeed(500)
    def on_init(self):
        ##load model 
        self.models = [] 
        for i in range(10):
            self.models.append(lgb.Booster(model_file="lightgbm_"+str(i)+".txt"))
        ##load bar to am 
        self.load_bar(2)
        self.last_price = self.am.close[-1] 
    def on_start(self):
        pass 
            
    def on_stop(self):
        pass
    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)
    def on_bar(self, bar: BarData):

        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return
        ##预测信号 
        self.last_price = am.close_array[-1] 
        self.unit = self.min_volume 
        self.sma_array = self.am.sma(self.sma_window,array = True) 
        self.indicator_array = get_bar_level_indicator_info(am,self.indicator_windows_list)
        self.indicator_info = np.array([item[-1] for item in self.indicator_array])
        predict =self.predict_siginal(self.indicator_info)

        if predict < 0.5:
            return 
        else:
            if self.sma_array[-1] > self.sma_array[-2]:##peak 
                if abs(self.pos) < self.min_volume:
                    self.short(self.last_price,self.unit) 
                else:
                    if self.pos > 0:
                        self.sell(self.last_price,abs(self.pos)) 
                    else:
                        self.short(self.last_price,self.unit) 
            else: ##bottom 
                if abs(self.pos) < self.min_volume:
                    self.buy(self.last_price,self.unit) 
                else:
                    if self.pos > 0: ##加多
                        self.buy(self.last_price,self.unit) 
                    else:  ##平空
                        self.cover(self.last_price,abs(self.pos))

    def predict_siginal(self,indicator_info) :
        predict=[]
        x=indicator_info.reshape(1,-1) 
        for model in self.models:
            predict.append(model.predict(x)[0])
        predict = np.mean(predict,axis=0)
        return predict 
    def on_trade(self, trade: TradeData):
        pass 

    def on_order(self, order: OrderData):
        pass 

    def on_stop_order(self, stop_order: StopOrder):
        pass

    
    
