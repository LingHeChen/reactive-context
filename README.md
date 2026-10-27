# Reactive Context

**让每一次模型请求，都与当前的资源、记忆和任务状态同步。**

Reactive Context 是一个 Python Agent 上下文中间件：在模型请求前核验来源版本、刷新受影响的依赖，并将选中的内容编译成不可变快照。通过 Pydantic AI 的请求边界接入，保留框架原有的 Agent 执行循环。

[在线演示](https://linghechen.github.io/reactive-context/) · [三来源示例](examples/three_parts/main.py) · [声明式 API 示例](examples/v2_dx.py) · [测试代码](tests)

Python **3.12+** · Pydantic **2** · 可选 Pydantic AI **2.60.0** · MIT

## 它解决什么问题

Agent 执行过程中，Skill 规则、工具目录和数据库结构可能发生变化；历史消息、长期事实和当前任务状态也有不同的生命周期。把它们拼成一段不断增长的 Prompt，很难解释当前请求到底使用了哪些信息、哪个版本。

本项目把这些输入变成可管理、可检查的上下文组成：

- **来源与依赖**：统一 Part 接口，显式依赖图，来源变化后刷新受影响的内容。
- **快照与编译**：记录内容、来源及版本，按权限、预算和稳定顺序构造请求。
- **记忆与状态**：区分原始历史、持久事实、工作目标和结构化工作状态。
- **受控更新**：Fact 入库检查，WorkState 字段级变更许可、版本检查和原子提交。
- **可解释性**：保留 manifest、trace 和来源身份，方便追踪实际入模内容。

```text
Skill / Tools / Schema / 选中的 Fact / 可选目标与状态
                         ↓
              版本核验 → 依赖刷新 → 快照 → Compiler
                                               ↓
当前任务 + 编译上下文 + 近期历史 / 按需恢复的原文 → 模型请求
```

## 快速开始

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后：

```bash
git clone https://github.com/LingHeChen/reactive-context.git
cd reactive-context
uv sync --frozen --all-extras

# 同一 Agent 会话：依次修改规则、切换工具、更新真实 SQLite 表结构
uv run python examples/three_parts/main.py

# 导出执行记录
uv run python examples/three_parts/main.py --json /tmp/three-parts.json

# state / computed / part 声明式上下文
uv run python examples/v2_dx.py
```

以上示例离线运行，不需要模型 API 密钥。三来源示例使用确定性 FunctionModel，文件读取和 SQLite 工具操作实际执行。

## 四类记忆如何分工

| 类型 | 保存什么 | 如何使用 |
| --- | --- | --- |
| **History** | 原始消息、工具调用与结果、归档引用 | 保留完整历史，提供近期投影，必要时按权限恢复原文 |
| **Fact** | 结构化内容、schema、证据、作用域、版本与生命周期 | 经检查后持久化，再按需选择或激活；保存不等于自动入模 |
| **Objective** | 目标、约束、成功条件、下一步、状态 | 可选工作目标组件，由宿主或受控工具更新 |
| **WorkState** | 宿主定义的业务状态、修订、字段变更许可和审计 | 可选结构化状态组件，未涉及字段默认保留 |

示例：[工作目标](examples/working_context/main.py) · [最小历史](examples/minimal_history/main.py) · [Fact 更新](examples/checked_fact_updates.py) · [工作状态](examples/structured_work_state.py) · [候选校验与提交](examples/checked_state_candidates.py)

## 本地预览项目页面

```bash
python3 -m http.server 4173 --bind 127.0.0.1 --directory site
```

打开 <http://127.0.0.1:4173>。页面为独立静态站，字体、GSAP 和演示素材均在仓库内；交互展示已保存的执行记录，不调用模型。`gh-pages` 分支仅包含网站发布产物。

## 开发与验证

```bash
uv sync --frozen --all-extras --group benchmark

# 部分离线测试依赖固定版本的 MemoryArena 解析器
# 缓存保存在本地，不随本仓库发布
git clone https://github.com/ZexueHe/MemoryArena.git .benchmark-cache/repos/memoryarena
git -C .benchmark-cache/repos/memoryarena checkout 6cd9de14b71915e39ac742a20dc33785e14b6aab

uv run pytest -q -W error::DeprecationWarning
uv run ruff check src tests examples benchmarks
uv run ruff format --check src tests examples benchmarks
uv run pyright
uv build
```

完整公开数据集未下载时，相关数据测试会跳过。上述命令执行代码检查和离线测试，不启动付费模型评测。

## 当前边界

- 本次公开源码来自本地冻结候选 `candidate-freeze-20261027`（`77fbe73`），公开仓库采用独立、干净的提交历史。
- 确定性测试和网页演示验证上下文机制，不证明真实模型任务成功率、记忆质量、成本下降或生产就绪。
- Fact 写入由宿主显式接入；历史检索提供恢复能力，但不能保证模型一定正确调用或使用结果。
- 来源权限、工具副作用和业务授权需要宿主正确配置；数据不能自行升级为系统指令。
- 库内缓存与模型服务端缓存是两层机制，稳定的请求顺序不保证服务端缓存命中率。

## 仓库结构

```text
src/reactive_context/   核心、编译器、来源、存储、记忆及框架适配
examples/              可运行示例
tests/                 单元、集成、属性与离线评测接口测试
benchmarks/            评测工具代码；执行前需自行配置数据与预算
site/                  项目展示页与必要素材
```

## License

项目代码采用 [MIT](LICENSE)。字体和页面依赖的许可保留在 [`site/assets/`](site/assets/) 中。
