目标与边界
•	目标：用 1,000 CAD 小资金，搭建可复制、可审计、可自动执行的“日/周级量化策略 + AI 过滤”的系统；先追求生存 + 纪律，再追求收益。
•	风险承受：你表示可承受 最大回撤 50%。本方案在配置上设定：
o	软阈值：-25%（触发降频/降仓/复盘）；
o	硬阈值：-50%（触发强制清仓 + 冷静期，人工复核后再开机）。
•	频率：日/周级，避免高频；现金账户优先，规避 PDT 约束（实际开户以你为准）。
•	执行市场：以美元计价的美国指数/行业 ETF** 为主（SPY/QQQ/IWM 等），必要时以美债/超短债 ETF 做防守；个股仅作辅助。
 
环境与依赖（先在 Mac 预研，再迁移到 4090 训练机）
macOS（Apple Silicon 或 Intel）
•	包管理：安装 Homebrew 与 miniforge（conda）。
•	Python 环境：conda create -n quant python=3.11 → conda activate quant。
•	科学栈：pip install pandas polars numpy scikit-learn xgboost catboost pyyaml loguru。
•	时序/回测：pip install vectorbtpro（或开源版 vectorbt）/backtrader（二选一）。
•	PyTorch：安装 MPS 版本（pip install torch torchvision torchaudio，参考官方指引）。
•	其他：brew install ta-lib 后 pip install TA-Lib（如需），pip install ib-insync optuna pytorch-lightning。
4090 训练机（Ubuntu 22.04）
•	安装 NVIDIA 驱动 + CUDA/cuDNN，匹配版本的 PyTorch (CUDA)。
•	追加：pip install flash-attn（可选，视CUDA/SM兼容而定）、pip install optuna 用于 HPO。
•	XGBoost GPU：pip install xgboost（官方轮子已支持 CUDA）。
实盘执行主要在 CPU 上；4090 主要用于训练/调参与较大 batch 的推理。
 
USD 与外汇/资金流（IBKR 实操要点）
•	资金与币种：账户可持多币种。若初始为 CAD，需在 IBKR 进行 CAD→USD 的外汇交易（CAD.USD 货币对，IDEALPRO）或使用自动货币转换；在成本模型中记录汇兑点差与佣金。
•	结算与PDT：美股T+1结算；若使用现金账户可规避 PDT 标记但需等待资金结算再复用；若用保证金账户需严格控制日内交易次数以免触发 PDT<25kUSD 限制。
•	回测处理：回测以 USD 为记账货币；若有跨币种资产，需以每日 USDCAD 汇率进行权益折算与成本叠加。
 
数据与成本建模
•	数据：日频为主；保存为 Parquet，按日期/标的分区。
•	成本：回测内置佣金 + 滑点（起步 10–20bps）；上线后用实盘成交对滑点回填校准。
•	T+1 约束：现金账户回测必须启用资金占用/释放时序，避免“资金重复使用”的假象。
 
监控与审计
•	日志：每笔订单/成交/PNL/触发的风控写入 CSV/SQLite。
•	可视化：日终输出权益曲线、回撤、胜率、换手、费用占比。
•	报警：日损、下单失败、数据断档、风控触发→邮件/Telegram。
 
推进计划（T+14 天上线，USD 版）
D0：在 Mac 完成环境搭建与仓库初始化，生成示例数据与报表模板。 D1–D2：拉取 USD ETF 日频数据（SPY/QQQ/IWM/XLK/XLE/XLV/TLT/SHY + 基准 BIL）；实现 features.py 与向量化回测雏形。 D3–D4：规则层（趋势/波动/成本门槛）跑通；做含成本的基础回测，输出权益/回撤/换手/费用占比。 D5–D6：在 Mac 上训练 TCN/TFT 小规模模型，接入 signal.py 胜率过滤；完成 walk-forward。 D7：在 IBKR 进行 CAD→USD 转换流程演练（Paper），验证记账与成本记录；完善 costs.py 的 FX 项。 D8–D9：迁移到 4090，开启 AMP + HPO（optuna 30 trials），保存最优权重与推理管线。 D10：接入 broker_ibkr.py（Paper）跑 信号→风控→下单→对账 全闭环。 D11–D12：影子周（Paper 全自动），修复异常、校准滑点/佣金/汇兑成本。 D13–D14：Go/No-Go 评审；若 Go，则以 100–200 USD/笔 极小仓位实盘。
 
Go-Live 审核清单（通过才允许上实盘）
☐	回测/影子周包含成本、T+1、失败重试、冷却期逻辑；
☐	单笔风险 ≤ 3%；单标 ≤ 20%；总敞口 ≤ 100%；
☐	日损/软DD/硬DD 能被自动识别并触发；
☐	下单失败/拒单/超时均会 停机 + 报警；
☐	配置/模型均有版本号，能回滚；
☐	演练过“断电/断线/券商维护”的人工切换流程。
 
日常SOP（实盘）
1.	开盘前：数据校验→生成当日候选→检查风控状态（是否处于冷却期）。
2.	开盘后：在设定时段运行 run_live.py；异常立停并报警。
3.	收盘后：对账→落库→输出日报（权益、回撤、费用、触发次数）。
4.	每周：参数/模型轻回顾（不频繁改动）；
5.	每月：策略复盘与风险审计（是否接近软/硬阈值）。
 
注意事项与红线
•	遵守硬阈值：达到 -50% 必须强制停机清仓，至少 10 个交易日冷静期；
•	不要 无上限加仓/马丁；
•	不要 把 LLM 直接接入生产改阈值；
•	先 Paper 后实盘，任何变更先影子运行一周；
•	备份：每日配置/模型/日志快照。
 
下一步
•	我可以基于此骨架，按你选定的 ETF 池与券商，生成可运行的最小代码模板（含示例数据与一键回测脚本）。
•	告诉我你想用的券商（IBKR/其他）、标的池（若有偏好）、以及是否用 XGBoost 还是 TCN/小型TFT，我就把 features.py / model.py / backtest.py / run_live.py 的初版代码填上。
<img width="432" height="647" alt="image" src="https://github.com/user-attachments/assets/000cc0ed-ec69-4e0f-94cb-01f3ef571045" />
