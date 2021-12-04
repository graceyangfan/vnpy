import numpy as np 
from vnpy.app.cta_strategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
)
from vnpy.trader.constant import (
    Status,
    Direction,
    Offset,
    Exchange
)
from vnpy.trader.utility import  round_to

from zenquant.feed.data import BarDataFeed 
from zenquant.env.observer import Observer 
from zenquant.utils.get_indicators_info import get_bar_level_indicator_info 


import torch
from elegantrl.agents.AgentDDPG import AgentDDPG
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.agents.AgentSAC import AgentSAC
from elegantrl.agents.AgentTD3 import AgentTD3
from elegantrl.agents.AgentA2C import AgentA2C
from elegantrl.train.config import Arguments
from elegantrl.train.run_tutorial import train_and_evaluate

MODELS = {"ddpg": AgentDDPG, "td3": AgentTD3, "sac": AgentSAC, "ppo": AgentPPO, "a2c": AgentA2C}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo", "a2c"]


class RLStrategy(CtaTemplate):
    ''''
    all trade volume should be positive 
    '''
    limit_total_margin_rate = 0.5  ##保证金和未成交订单的最大占用比率
    limit_order_margin_rate = 0.5  ##每次下单最大占用比率
    parameters = ["limit_total_margin_rate","limit_order_margin_rate"]
    variables = []
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.vt_symbol = vt_symbol 

        ## params  for portfolio 
        self.MarginLevel = setting["MarginLevel"] 
        self.start_margin_rate = 1.0/self.MarginLevel 
        ##paramters for strategy 
        self.windows_size = 1 
        self.pos_info_scale  = setting["pos_info_scale"]
        self.indicator_info_scale = setting["indicator_info_scale"] 
        #
        self.indicator_windows_list = setting["indicator_windows_list"] 
        ## agent 
        self.agent_name = setting['agent_name']
        self.net_dim = setting['net_dim']
        self.state_dim = setting['state_dim'] 
        self.action_dim = setting['action_dim'] 
        self.agent = MODELS[self.agent_name]()
        self.cwd = setting['cwd'] 
        self.agent.init(self.net_dim, self.state_dim, self.action_dim)
        self.agent.save_or_load_agent(cwd=self.cwd, if_save=False) 
        self.act = self.agent.act
        self.device = self.agent.device
        ##use for default 
        self.balance = 0.0
        self.frozen = 0.0 
        self.available = 0.0 
        self.hold_pos = 0.0 
        self.pos_avgprice = 0.0 
        self.pnl = 0.0 

        ##初始化数据
        self.observer = Observer(self.state_dim,self.windows_size)
        self.bg = BarGenerator(self.on_bar) 
        self.am = BarDataFeed(500)
        
    def on_init(self):
        self.pricetick = self.get_pricetick()
        self.min_volume = self.get_min_volume()
    def on_start(self):
        pass  
            
    def on_stop(self):
        pass
    def on_tick(self, tick: TickData):
        ##不依赖于polifo的计算，直接从交易所获得数据
        ##实时更新pos_info 和资产 
        self.last_price = tick.last_price   

    def on_bar(self, bar: BarData):

        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return

        ## update Pos Info 
        self.active_orders = self.get_active_orders() 
        frozen_occupy_margin = sum([order.price * abs(order.volume) * self.start_margin_rate for order in self.active_orders])
        self.total_occupy_margin = frozen_occupy_margin + abs(self.hold_pos)*self.pos_avgprice * self.start_margin_rate
        self.available = self.balance - self.total_occupy_margin -self.frozen
        self.capital = self.balance  +  self.pnl 
        self.occupy_rate = self.total_occupy_margin / self.capital 
        self.pos_info = np.array([self.occupy_rate,
        abs(self.hold_pos)>self.min_volume,
        1.0-self.pos_avgprice/self.last_price])
        self.pos_info = self.pos_info * self.pos_info_scale 

        ##update bar level info 
        self.indicator_array = get_bar_level_indicator_info(am,self.indicator_windows_list)
        self.indicator_info = np.array([item[-1] for item in self.indicator_array])
        self.indicator_info = self.indicator_info * self.indicator_info_scale
        obs=self.observer.observe(self.indicator_info,self.pos_info).reshape((-1,))
        ##get action 
        s_tensor = torch.as_tensor((obs,), device=self.device)
        s_tensor = s_tensor.float()
        a_tensor = self.act(s_tensor)  
        action = (
                    a_tensor.detach().cpu().numpy()[0]
                ) 
        ## take action 
        self.create_order(action[0])
        self.put_event() 


    def on_trade(self, trade: TradeData):
        pass 

    def on_order(self, order: OrderData):
        pass 

    def on_stop_order(self, stop_order: StopOrder):
        pass

    def on_account(self,account):  
        self.account = account

        if account.accountid == "USDT":
            self.balance = account.balance  ##所有资金
            self.frozen =  account.frozen   ##维系保证金 

    def on_position(self, position):
        self.position = position 
        self.hold_pos = position.volume 
        self.pnl = position.pnl 
        if abs(self.hold_pos) <self.min_volume:
            self.pos_avgprice = self.last_price
        else:
            self.pos_avgprice = position.price  

    def create_order(self,action):
        direction = None
        offset = None 
        if abs(self.hold_pos) > self.min_volume:
            if self.hold_pos > 0: ##多仓 
                if action > 0:
                    direction = Direction.LONG
                    offset = Offset.OPEN 
                else:
                    direction = Direction.SHORT
                    offset = Offset.CLOSE
            else:  #空仓 
                if action > 0: ##平空
                    direction = Direction.LONG
                    offset = Offset.CLOSE 
                else: #开空
                    direction = Direction.SHORT
                    offset = Offset.OPEN  
        else:
            ##没有持仓
            if action>0:
                direction = Direction.LONG
                offset = Offset.OPEN 
            else:
                direction = Direction.SHORT
                offset = Offset.OPEN 
        trade_ptr = abs(action) 

        ##检查当前中保证金占用率,是否超过限制，可平仓不可开仓
        if self.occupy_rate > self.limit_total_margin_rate:
           # self.output("total margin limited triggered")
            if offset == Offset.OPEN:
                return None
                ##开仓
        
        if offset == Offset.OPEN:
            ##限制每一笔的最大金额比例
            trade_balance = self.available*self.limit_order_margin_rate * trade_ptr
            self.min_trade_in_gateway= self.min_volume * self.last_price /self.MarginLevel 
            trade_price = round_to(self.last_price,self.pricetick)
            trade_volume = round_to(trade_balance * self.MarginLevel /trade_price,self.min_volume)
            if trade_balance < self.min_trade_in_gateway:
                return None 
        ##平仓
        elif offset == Offset.CLOSE:
            max_trade_volume = abs(self.hold_pos)
            trade_price = round_to(self.last_price,self.pricetick)
            trade_volume = round_to(max_trade_volume * trade_ptr,self.min_volume)
            if trade_volume < self.min_volume:
                #self.output("trade volume is not enough to  close a order")
                return None 


        return self.send_order(
            direction,
            offset,
            trade_price,
            abs(trade_volume),
            False,
            False,
            False 
        )
