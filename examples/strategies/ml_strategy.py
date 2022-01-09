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
    loss_stop_rate = -0.1
    indicator_windows_list= [6,12,48,168]
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
        self.indicator_windows_list = setting["indicator_windows_list"] 
        self.start_margin_rate = 1.0/self.MarginLevel 
        self.capital = 100 
        self.init_capital = setting['init_capital'] 
        self.capital = self.init_capital
        self.balance = 1e-3
        self.frozen = 1e-3
        self.available = 1e-3
        self.hold_pos = 0
        self.pos_avgprice = 1e-3
        self.pnl = 1e-3
        self.last_price = 1e-3
        self.account_ready = False
        self.position_ready = False 
        self.history_action = 0 
        self.history_pnl = 0 

        self.bg = BarGenerator(self.on_bar) 
        self.am = BarDataFeed(500)
    def on_init(self):
        ##load model 
        self.models = [] 
        for i in range(5):
            self.models.append(lgb.Booster(model_file="lightgbm_"+str(i)+".txt"))
        ##load bar to am 
        self.pricetick = self.get_pricetick()
        self.min_volume = self.get_min_volume()
        self.load_bar(2)
        self.last_price = self.am.close[-1] 
    def on_start(self):
        pass 
            
    def on_stop(self):
        pass
    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)
        self.last_price = tick.last_price 

    def on_bar(self, bar: BarData):

        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return

        if not self.account_ready:
            return
        if not self.position_ready:
            return

            
        ## update Pos Info 
        self.active_orders = self.get_active_orders().values()

        frozen_occupy_margin = sum([order.price * abs(order.volume) * self.start_margin_rate for order in self.active_orders])
        self.pos_occupy_margin =  abs(self.hold_pos)*self.pos_avgprice * self.start_margin_rate
        self.total_occupy_margin = frozen_occupy_margin +self.pos_occupy_margin 
        self.available = self.balance - self.total_occupy_margin -self.frozen
        self.last_capital = self.capital 
        self.capital = self.balance  +  self.pnl 
        self.occupy_rate = self.total_occupy_margin /  self.capital


        ##stop loss and profit 
        if abs(self.hold_pos) > self.min_volume:
            if self.pnl /self.pos_occupy_margin > self.profit_stop_rate or \
                self.pnl /self.pos_occupy_margin < self.loss_stop_rate: 
                if self.hold_pos >0:
                    self.send_order(
                        Direction.SHORT,
                        Offset.CLOSE,
                        self.last_price,
                        abs(self.hold_pos),
                        False,
                        False)
                else:
                    self.send_order(
                        Direction.LONG,
                        Offset.CLOSE,
                        self.last_price,
                        abs(self.hold_pos),
                        False,
                        False)

        #self.atr= self.am.atr(self.atr_window,array = False)
        #self.unit = self.available*self.available_change_percent*self.MarginLevel/self.atr
        self.unit = 1 
        self.unit = round_to(self.unit*self.min_volume,self.min_volume)
        self.indicator_array = get_bar_level_indicator_info(am,self.indicator_windows_list)
        self.indicator_info = np.array([item[-1] for item in self.indicator_array])
        siginal =self.predict_siginal(self.indicator_info)


        ##限制开单
        if self.occupy_rate > self.limit_total_margin_rate:
            return None 

        if siginal == 0:
            return None  
        else:
            if siginal == 2: ##peak  
                if abs(self.hold_pos) < self.min_volume:
                    self.short(self.last_price,self.unit) 
                else:
                    if self.hold_pos > 0:
                        self.sell(self.last_price,abs(self.hold_pos)) 
                    else:
                        self.short(self.last_price,self.unit) 
            else: ##bottom 
                if abs(self.hold_pos) < self.min_volume:
                    self.buy(self.last_price,self.unit) 
                else:
                    if self.hold_pos > 0: ##加多
                        self.buy(self.last_price,self.unit) 
                    else:  ##平空
                        self.cover(self.last_price,abs(self.hold_pos))

    def predict_siginal(self,indicator_info) :
        predict=[]
        x=indicator_info.reshape(1,-1) 
        for model in self.models:
            predict.append(model.predict(x))
        prob=np.mean(predict,axis=0)
        prob_max =np.max(prob)
        ##prob_max to small no trade 
        if prob_max>0.5:
            siginal = np.argmax(prob)
        else:
            siginal =0 
        return siginal 

    def on_trade(self, trade: TradeData):
        pass 

    def on_order(self, order: OrderData):
        pass 

    def on_stop_order(self, stop_order: StopOrder):
        pass

    def on_account(self,account):  
            
        self.account = account

        if account.accountid == "USDT":
            self.account_ready = True 
            self.balance = account.balance  ##所有资金
            self.frozen =  account.frozen   ##维系保证金 

    def on_position(self, position):

        if position:
            self.position_ready = True 
        self.position = position 
        self.hold_pos = position.volume 
        self.pnl = position.pnl 
        if abs(self.hold_pos) <self.min_volume:
            self.pos_avgprice = self.last_price
        else:
            self.pos_avgprice = position.price
    
    
