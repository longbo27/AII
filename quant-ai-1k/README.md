# quant-ai-1k

`quant-ai-1k` 是一个面向 D0 交付的量化策略项目骨架，覆盖数据抓取、特征工程、回测执行与基础报表输出。项目以 Python 3.11 为基础，支持 macOS（Apple Silicon/Intel）CPU 环境运行，并预留未来向 GPU/CUDA 迁移的容器模板。

## 功能概览
- **数据获取**：基于 `yfinance` 按配置拉取 ETF/债券日线行情。
- **特征工程**：纯 pandas/numpy 计算收益率、RSI、MACD、ATR 等基础指标。
- **策略回测**：趋势 + 波动过滤、等权调仓、单标权重上限、T+1 执行与滑点成本。
- **报表输出**：输出回测时序、权益曲线图与摘要指标。
- **配置化风险控制**：通过 YAML 配置最大回撤、仓位、日损等约束。

## 目录结构
```
quant-ai-1k/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── backtest.yml
│   ├── risk.yml
│   └── universe.yml
├── data/
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
├── reports/
│   └── .gitkeep
├── scripts/
│   ├── fetch_data.py
│   └── run_backtest.py
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── features.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── risk.py
│   └── backtest/
│       ├── __init__.py
│       └── engine.py
└── docker/
    ├── Dockerfile.gpu
    └── docker-compose.gpu.yml
```

## 环境准备
### 方案一：Python 内置 venv
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 方案二：micromamba（可选）
```bash
micromamba create -n quant-ai-1k python=3.11 -y
micromamba activate quant-ai-1k
pip install -r requirements.txt
```

## 本地运行
```bash
# 1. 拉取行情（默认从 2010-01-01 开始）
python scripts/fetch_data.py

# 2. 运行回测
python scripts/run_backtest.py
```

运行成功后，`reports/` 目录会生成：
- `backtest_timeseries.csv`
- `equity_curve.png`
- `summary.json`

若 `summary.json` 中的关键指标（`total_return`, `max_drawdown`, `daily_vol`, `sharpe`）均非 `NaN`，则视为流程通过。

## GPU/CUDA 迁移（预留）
- `docker/Dockerfile.gpu`：面向 Linux + NVIDIA 环境的基础镜像模板。
- `docker/docker-compose.gpu.yml`：单服务占位配置，未来可扩展挂载数据与 GPU 运行。

当前在 macOS/CPU 环境无需执行 Docker 相关命令。待迁移至 GPU 时，可在 Linux 主机上修改镜像标签并运行：
```bash
docker compose -f docker/docker-compose.gpu.yml up --build
```

## 风险提示
本项目仅为策略研究骨架，未覆盖真实交易所需的风控与风控审批流程。投入真实资金前需进行充分验证与合规审查。
