import numpy as np 
from scipy import special
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
from tzlocal import get_localzone
from  datetime  import datetime
from zenquant.trader.utility import  round_to

from zenquant.feed.data import BarDataFeed 
from zenquant.env.observer import Observer 
from zenquant.utils.get_indicators_info import get_bar_level_indicator_info 
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-6)

LOCAL_TZ = get_localzone()
class MLStrategy(CtaTemplate):
    limit_total_margin_rate = 0.3  ##保证金和未成交订单的最大占用比率
    available_change_percent= 0.05 ##atr use percent 
    atr_window = 20 
    atr_dev = 2.0 
    sma_window = 10 
    profit_stop_rate = 1.0
    loss_stop_rate = -0.1
    indicator_windows_list= [6,12,48,168]
    parameters = [ 
        "limit_total_margin_rate",
        "available_change_percent",
        "atr_window",
        "atr_dev",
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
        self.profit_stop_rate = setting["profit_stop_rate"] 
        self.loss_stop_rate = setting["loss_stop_rate"] 
        self.available_change_percent = setting["available_change_percent"] 
        self.atr_window = setting["atr_window"] 
        self.atr_dev = setting["atr_dev"] 
        #self.sma_window = setting["sma_window"] 
        ##马丁参数 
        self.increase_pos_ptr = setting["increase_pos_ptr"]
        self.trading_value_multiplier = setting["trading_value_multiplier"] 
        self.martingle_init_pos = setting["martingle_init_pos"] 
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
        self.capital = 100 
        self.init_capital = setting['init_capital'] 
        self.capital = self.init_capital
        self.atr = 0 
        self.last_order_time =datetime.now(LOCAL_TZ)
        self.inverse_stop_open = False 

        self.bg = BarGenerator(self.on_bar) 
        self.am = BarDataFeed(800)
    def on_init(self):
        ##load model 
        self.models = [] 
        for i in range(5):
            self.models.append(lgb.Booster(model_file="lightgbmeth_"+str(i)+".txt"))
        ##load bar to am 
        self.pricetick = self.get_pricetick()
        self.min_volume = self.get_min_volume()
        print(self.min_volume)
        self.load_bar(2)
        self.last_price = self.am.close[-1] 
    def on_start(self):
        pass 
            
    def on_stop(self):
        pass
    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)
        self.last_price = tick.last_price
        self.long_cross_price = tick.ask_price_1
        self.short_cross_price = tick.bid_price_1

        ##stop loss and profit 
        if abs(self.hold_pos) > self.min_volume:
            pnl = (self.last_price - self.pos_avgprice)*self.hold_pos 
            margin = abs(self.hold_pos)*self.pos_avgprice*self.start_margin_rate 
            ##stop profit 
            if self.inverse_stop_open and self.hold_pos > 0:
                if self.last_price - self.pos_avgprice > self.atr_dev * self.atr:
                    self.sell(self.short_cross_price , abs(self.hold_pos))
                    self.write_log(f"马丁止盈{self.short_cross_price}@{abs(self.hold_pos)}盈利{pnl}")
                    self.inverse_stop_open = False 
            elif self.inverse_stop_open and self.hold_pos < 0:
                if self.pos_avgprice - self.last_price > self.atr_dev*self.atr:
                    self.cover(self.long_cross_price, abs(self.hold_pos)) 
                    self.write_log(f"马丁止盈{self.long_cross_price}@{abs(self.hold_pos)}盈利{pnl}")
                    self.inverse_stop_open = False 
            else:
                if pnl / margin > self.profit_stop_rate:
                    self.inverse_stop_open = False  
                    if self.hold_pos >0:
                        self.sell(self.short_cross_price , abs(self.hold_pos))
                        self.write_log(f"止盈{self.short_cross_price}@{abs(self.hold_pos)}盈利{pnl}")
                    else:
                        self.cover(self.long_cross_price, abs(self.hold_pos)) 
                        self.write_log(f"止盈{self.long_cross_price}@{abs(self.hold_pos)}盈利{pnl}")
            ## stop loss 
            if pnl / margin < self.loss_stop_rate: 
                self.inverse_stop_open = True
                if self.hold_pos >0: 
                    self.sell(self.short_cross_price , abs(self.hold_pos))
                    self.write_log(f"止损{self.short_cross_price}@{abs(self.hold_pos)}盈利{pnl}")
                    self.prev_enter_price = self.short_cross_price 
                    self.prev_enter_pos =round_to(self.martingle_init_pos*self.min_volume,self.min_volume)
                    self.short(self.prev_enter_price, self.prev_enter_pos) 
                    self.write_log(f"止损反转开空{ self.prev_enter_pos}@{self.prev_enter_price}")
                else:
                    self.cover(self.long_cross_price, abs(self.hold_pos))
                    self.write_log(f"止损{self.long_cross_price}@{abs(self.hold_pos)}盈利{pnl}") 
                    self.prev_enter_price = self.long_cross_price
                    self.prev_enter_pos =round_to(self.martingle_init_pos*self.min_volume,self.min_volume)
                    self.buy(self.prev_enter_price, self.prev_enter_pos) 
                    self.write_log(f"止损反转开空{ self.prev_enter_pos}@{self.prev_enter_price}")

    def on_bar(self, bar: BarData):
        self.cancel_all()
        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return

        if not self.account_ready:
            return
        if not self.position_ready:
            return
        self.write_log(f"持仓{self.hold_pos}余额{self.capital}")
            
        ## update Pos Info 
        self.active_orders = self.get_active_orders().values()

        frozen_occupy_margin = sum([order.price * abs(order.volume) * self.start_margin_rate for order in self.active_orders])
        self.pos_occupy_margin =  abs(self.hold_pos)*self.pos_avgprice * self.start_margin_rate
        self.total_occupy_margin = frozen_occupy_margin +self.pos_occupy_margin 
        self.available = self.balance - self.total_occupy_margin -self.frozen
        self.last_capital = self.capital 
        self.capital = self.balance  +  self.pnl 
        self.occupy_rate = self.total_occupy_margin /  self.capital

       
       ##检查当前模式
        if self.inverse_stop_open and abs(self.hold_pos) > self.min_volume:
            ##进入马丁策略
            self.martingle()
        else:
            self.atr= self.am.atr(self.atr_window,array = False)
            # self.sma_array = self.am.sma(self.sma_window,array = True) 
            self.unit = self.available*self.available_change_percent*self.MarginLevel/self.atr
            #self.unit = 1 
            self.unit = round_to(self.unit*self.min_volume,self.min_volume)
            self.indicator_array = get_bar_level_indicator_info(am,self.indicator_windows_list)
            self.indicator_info = np.array([item[-1] for item in self.indicator_array])
            self.ml_decision()

    def martingle(self):
        ##被止损 或止盈 退出马丁模式 
        if self.hold_pos > 0: ##当前持有多仓 
            down_percent = (self.prev_enter_price-self.last_price)/self.prev_enter_price
            if self.occupy_rate < self.limit_total_margin_rate and down_percent > self.increase_pos_ptr:
                self.buy(self.long_cross_price,self.trading_value_multiplier * self.prev_enter_pos)
                self.prev_enter_price = self.long_cross_price
                self.prev_enter_pos = self.trading_value_multiplier * self.prev_enter_pos
                self.write_log(f"马丁开多{self.prev_enter_pos }@{self.long_cross_price}")
                
        else:
            down_percent = (self.last_price - self.prev_enter_price)/self.prev_enter_price
            if self.occupy_rate < self.last_toatal_margin_rate and down_percent > self.increase_pos_ptr:
                self.short(self.short_cross_price ,self.trading_value_multiplier * self.prev_enter_pos)
                self.prev_enter_price = self.short_cross_price 
                self.prev_enter_pos = self.trading_value_multiplier * self.prev_enter_pos
                self.write_log(f"马丁开空{self.prev_enter_pos}@{self.short_cross_price }")
    
    def ml_decision(self):
        ##开始检查是否开单 
        ##限制开单
        if self.occupy_rate > self.limit_total_margin_rate:
            return None 
        order_gap = (datetime.now(LOCAL_TZ) - self.last_order_time).seconds
        open_order = False
        if order_gap < 120:##两次开仓需大于2min 
            return None 
    
        ##预测信号，开干
        signal = self.predict_siginal(self.indicator_info)
        if signal == 0 :
            return None 
        else:
            if signal == 2:
                if self.hold_pos > self.min_volume:
                    self.sell(self.short_cross_price ,abs(self.hold_pos))
                    self.write_log(f"平多{abs(self.hold_pos)}@{self.short_cross_price }")
                self.short(self.short_cross_price ,self.unit)
                self.last_order_time = datetime.now(LOCAL_TZ)
                self.write_log(f"开空{self.unit}@{self.short_cross_price}")
            else:
                if self.hold_pos < -self.min_volume:
                    self.cover(self.long_cross_price,abs(self.hold_pos))
                    self.write_log(f"平空{abs(self.hold_pos)}@{self.long_cross_price}")
                self.buy(self.long_cross_price,self.unit)
                self.last_order_time = datetime.now(LOCAL_TZ)
                self.write_log(f"开多{self.unit}@{self.long_cross_price}")

    def predict_siginal(self,indicator_info) :
        predict=[]
        x=indicator_info.reshape(1,-1) 
        for model in self.models:
            predict.append(np.argmax(softmax(model.predict(x))))
        
        return  np.argmax(np.bincount(predict))

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
    
    
